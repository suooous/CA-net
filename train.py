import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CANet
from dataset import SimpleSegDataset
from losses import multi_class_segmentation_loss, contour_completion_loss
from utils import mask_to_boundary, save_checkpoint
import torch.optim as optim
import numpy as np

def train_one_epoch(model, loader, opt, device, alpha=1.0):
    model.train()
    total_loss = 0.0
    seg_loss_total = 0.0
    contour_loss_total = 0.0
    
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        preds, contour_pred = model(imgs)
        
        # ================= 分割损失（多尺度） =================
        seg_loss = multi_class_segmentation_loss(preds, masks, alpha=0.5)
        
        # ================= 生成轮廓标签 =================
        bgt_list = []
        for m in masks.cpu().numpy():
            binary_mask = (m > 0).astype(np.uint8)  # 前景/背景
            b = mask_to_boundary(binary_mask)
            bgt_list.append(b)
        bgt = torch.tensor(np.stack(bgt_list, axis=0), device=device).unsqueeze(1).float()  # [B,1,H,W]
        
        if contour_pred.shape[-2:] != bgt.shape[-2:]:
            bgt = F.interpolate(bgt, size=contour_pred.shape[-2:], mode='bilinear', align_corners=False)
        
        # ================= 轮廓补全损失 =================
        contour_loss = contour_completion_loss(contour_pred, bgt)
        
        # ================= 总损失 =================
        loss = seg_loss + alpha * contour_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        seg_loss_total += seg_loss.item()
        contour_loss_total += contour_loss.item()
    
    return total_loss / len(loader), seg_loss_total / len(loader), contour_loss_total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    eps = 1e-6
    sum_dice, sum_iou, sum_acc, sum_bf1 = 0.0, 0.0, 0.0, 0.0
    count = 0
    
    # 多类别评估指标
    class_dice = {0: 0.0, 1: 0.0, 2: 0.0}
    class_iou = {0: 0.0, 1: 0.0, 2: 0.0}
    class_count = {0: 0, 1: 0, 2: 0}
    
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds, contour_pred = model(imgs)
        p = preds[-1]
        if p.shape[-2:] != masks.shape[-2:]:
            p = F.interpolate(p, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        
        # 多类别预测
        pred_probs = torch.softmax(p, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)
        
        # 类别级指标
        for class_id in range(3):
            pred_mask = (pred_classes == class_id).float()
            true_mask = (masks == class_id).float()
            
            if torch.sum(true_mask) > 0:
                inter = (pred_mask * true_mask).sum(dim=(1,2))
                pred_sum = pred_mask.sum(dim=(1,2))
                targ_sum = true_mask.sum(dim=(1,2))
                union = pred_sum + targ_sum - inter
                
                dice = (2*inter + eps) / (pred_sum + targ_sum + eps)
                iou = (inter + eps) / (union + eps)
                
                class_dice[class_id] += float(dice.mean().item())
                class_iou[class_id] += float(iou.mean().item())
                class_count[class_id] += 1
        
        # 整体指标（前景/背景二值）
        pred_bin = (pred_classes > 0).float()
        target = (masks > 0).float()
        inter = (pred_bin * target).sum(dim=(1,2))
        pred_sum = pred_bin.sum(dim=(1,2))
        targ_sum = target.sum(dim=(1,2))
        union = pred_sum + targ_sum - inter
        dice = (2*inter + eps) / (pred_sum + targ_sum + eps)
        iou = (inter + eps) / (union + eps)
        acc = ((pred_bin == target).float().mean(dim=(1,2)))
        
        # Boundary F1
        bgt_list = []
        for m in masks.cpu().numpy():
            binary_mask = (m > 0).astype(np.uint8)
            b = mask_to_boundary(binary_mask)
            bgt_list.append(b)
        bgt = torch.tensor(np.stack(bgt_list, axis=0), device=device).unsqueeze(1).float().to(device)
        if contour_pred.shape[-2:] != bgt.shape[-2:]:
            bgt = F.interpolate(bgt, size=contour_pred.shape[-2:], mode='bilinear', align_corners=False)
        bpred_bin = (contour_pred >= 0.5).float()
        
        tp = (bpred_bin * (bgt > 0.5).float()).sum(dim=(1,2,3))
        fp = (bpred_bin * (1.0 - (bgt > 0.5).float())).sum(dim=(1,2,3))
        fn = (((1.0 - bpred_bin) * (bgt > 0.5).float())).sum(dim=(1,2,3))
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)
        bf1 = (2 * prec * rec) / (prec + rec + eps)
        
        sum_dice += float(dice.mean().item())
        sum_iou += float(iou.mean().item())
        sum_acc += float(acc.mean().item())
        sum_bf1 += float(bf1.mean().item())
        count += 1
    
    # 平均每类指标
    for class_id in range(3):
        if class_count[class_id] > 0:
            class_dice[class_id] /= class_count[class_id]
            class_iou[class_id] /= class_count[class_id]
    
    return {
        'dice': sum_dice / max(1, count),
        'iou': sum_iou / max(1, count),
        'acc': sum_acc / max(1, count),
        'boundary_f1': sum_bf1 / max(1, count),
        'class_dice': class_dice,
        'class_iou': class_iou
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集路径
    train_img = './data/train_image'
    train_mask = './data/train_label'
    test_img = './data/test_image'
    test_mask = './data/test_label'
    
    if not os.path.exists(train_img):
        print(f"❌ 训练图像目录不存在: {train_img}")
        return
    if not os.path.exists(train_mask):
        print(f"❌ 训练标签目录不存在: {train_mask}")
        return
    
    # 数据加载
    ds = SimpleSegDataset(train_img, train_mask, num_classes=3)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    test_ds = SimpleSegDataset(test_img, test_mask, num_classes=3)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"✅ 数据集加载成功")
    print(f"   训练集大小: {len(ds)}")
    print(f"   测试集大小: {len(test_ds)}")
    
    # 模型初始化
    model = CANet(pretrained=True, num_classes=3).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"✅ 模型初始化成功")
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    for epoch in range(1, 51):
        total_loss, seg_loss, contour_loss = train_one_epoch(model, loader, opt, device, alpha=1.0)
        print(f'Epoch {epoch:2d} - 总损失: {total_loss:.4f} | 分割损失: {seg_loss:.4f} | 轮廓损失: {contour_loss:.4f}')
        
        if epoch % 5 == 0:
            metrics = evaluate(model, test_loader, device)
            print(f"Eval — Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | Acc: {metrics['acc']:.4f} | bF1: {metrics['boundary_f1']:.4f}")
            print("类别指标:")
            for class_id in range(3):
                if metrics['class_dice'][class_id] > 0:
                    print(f"  类别 {class_id}: Dice={metrics['class_dice'][class_id]:.4f}, IoU={metrics['class_iou'][class_id]:.4f}")
        
        # 保存
        print(f"🔄 正在保存检查点...")
        save_checkpoint({'model':model.state_dict(), 'opt':opt.state_dict(), 'epoch':epoch},
                        f'checkpoints/epoch_{epoch}.pth')
        
        if epoch % 10 == 0:
            print(f"🔄 正在保存最佳模型...")
            save_checkpoint({'model':model.state_dict(), 'opt':opt.state_dict(), 'epoch':epoch},
                            f'checkpoints/best_model_epoch_{epoch}.pth')


if __name__ == '__main__':
    main()
