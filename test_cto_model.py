# %%
import os
import numpy as np
from glob import glob
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import SimpleSegDataset
import matplotlib.pyplot as plt
import torch
from losses import DiceCoefficient, DiceLoss
import torch.optim as optim
from model import CANet
from PIL import Image
import torch.nn.functional as F
from metric import dice_score, iou_score

def evaluate_canet_model(model, loader, device, num_classes=3):
    """评估CA-net模型的性能"""
    model.eval()
    dice_sum = torch.zeros(num_classes, device=device)
    iou_sum  = torch.zeros(num_classes, device=device)
    dice_mean_sum = 0.0
    iou_mean_sum  = 0.0
    n = 0
    
    # 轮廓检测指标
    contour_dice_sum = 0.0
    contour_iou_sum = 0.0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds, contour_pred = model(imgs)
            
            # 使用最后一个预测结果进行评估
            p = preds[-1]  # 最终输出
            
            # 确保预测和标签尺寸匹配
            if p.shape[-2:] != lbls.shape[-2:]:
                p = F.interpolate(p, size=lbls.shape[-2:], mode='bilinear', align_corners=False)
            
            # 计算分割指标
            dice_pc, dice_m = dice_score(p, lbls, num_classes)
            iou_pc,  iou_m  = iou_score(p, lbls, num_classes)

            dice_sum += dice_pc
            iou_sum  += iou_pc
            dice_mean_sum += dice_m
            iou_mean_sum  += iou_m
            
            # 计算轮廓检测指标
            # 生成轮廓标签
            bgt_list = []
            for m in lbls.cpu().numpy():
                binary_mask = (m > 0).astype(np.uint8)  # 前景/背景
                from utils import mask_to_boundary
                b = mask_to_boundary(binary_mask)
                bgt_list.append(b)
            
            bgt = torch.tensor(np.stack(bgt_list, axis=0), device=device).unsqueeze(1).float()
            
            # 确保轮廓预测和标签尺寸匹配
            if contour_pred.shape[-2:] != bgt.shape[-2:]:
                contour_pred_resized = F.interpolate(contour_pred, size=bgt.shape[-2:], mode='bilinear', align_corners=False)
            else:
                contour_pred_resized = contour_pred
            
            # 轮廓二值化
            contour_pred_bin = (contour_pred_resized >= 0.5).float()
            
            # 轮廓Dice和IoU
            eps = 1e-6
            inter = (contour_pred_bin * bgt).sum(dim=(1,2,3))
            pred_sum = contour_pred_bin.sum(dim=(1,2,3))
            targ_sum = bgt.sum(dim=(1,2,3))
            union = pred_sum + targ_sum - inter
            
            contour_dice = (2*inter + eps) / (pred_sum + targ_sum + eps)
            contour_iou = (inter + eps) / (union + eps)
            
            contour_dice_sum += float(contour_dice.mean().item())
            contour_iou_sum += float(contour_iou.mean().item())
            
            n += 1

    # 计算平均值
    dice_per_class = (dice_sum / n).cpu().tolist()
    iou_per_class  = (iou_sum  / n).cpu().tolist()
    dice_mean      = (dice_mean_sum / n).item()
    iou_mean       = (iou_mean_sum  / n).item()
    contour_dice_mean = contour_dice_sum / n
    contour_iou_mean = contour_iou_sum / n

    print(f"=== CA-net模型性能评估结果 ===")
    print(f"测试样本数量: {n}")
    print(f"\n分割性能:")
    print(f"Dice per class: {dice_per_class}")
    print(f"Mean  Dice   : {dice_mean:.4f}")
    print(f"IoU  per class: {iou_per_class}")
    print(f"Mean  IoU    : {iou_mean:.4f}")
    print(f"\n轮廓检测性能:")
    print(f"Contour Dice : {contour_dice_mean:.4f}")
    print(f"Contour IoU  : {contour_iou_mean:.4f}")

    return {
        'dice_per_class': dice_per_class,
        'dice_mean': dice_mean,
        'iou_per_class': iou_per_class,
        'iou_mean': iou_mean,
        'contour_dice': contour_dice_mean,
        'contour_iou': contour_iou_mean
    }

def visualize_test_results(model, test_loader, device, num_samples=5):
    """可视化测试结果"""
    model.eval()
    
    # 创建子图
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds, contour_pred = model(imgs)
            
            # 获取最终预测
            p = preds[-1]
            if p.shape[-2:] != lbls.shape[-2:]:
                p = F.interpolate(p, size=lbls.shape[-2:], mode='bilinear', align_corners=False)
            
            # 转换为numpy
            img_np = imgs[0, 0].cpu().numpy()  # 灰度图
            lbl_np = lbls[0].cpu().numpy()
            pred_np = torch.softmax(p, dim=1)[0].cpu().numpy()
            contour_np = contour_pred[0, 0].cpu().numpy()
            
            # 预测类别
            pred_class = np.argmax(pred_np, axis=0)
            
            # 显示结果
            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f'原始图像 {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(lbl_np, cmap='tab10', vmin=0, vmax=2)
            axes[i, 1].set_title(f'真实标签 {i+1}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_class, cmap='tab10', vmin=0, vmax=2)
            axes[i, 2].set_title(f'预测结果 {i+1}')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(contour_np, cmap='hot')
            axes[i, 3].set_title(f'轮廓预测 {i+1}')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('canet_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("测试结果可视化已保存到: canet_test_results.png")

# %%
def main():
    # 数据集路径
    data_root = './data'
    test_img = os.path.join(data_root, 'test_image')
    test_lbl = os.path.join(data_root, 'test_label')
    
    # 检查路径
    if not os.path.exists(test_img):
        print(f"❌ 测试图像目录不存在: {test_img}")
        return
    if not os.path.exists(test_lbl):
        print(f"❌ 测试标签目录不存在: {test_lbl}")
        return
    
    # 创建测试数据集
    test_dataset = SimpleSegDataset(test_img, test_lbl, num_classes=3)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"✅ 测试数据集加载成功")
    print(f"   测试集大小: {len(test_dataset)}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = CANet(pretrained=False, num_classes=3).to(device)
    
    # 检查点路径
    checkpoint_path = 'checkpoints/epoch_50.pth'  # 可以根据实际情况调整
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        print("请确保已训练模型并保存检查点")
        return
    
    # 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✅ 模型加载成功: {checkpoint_path}")
    print(f"   训练轮次: {checkpoint.get('epoch', 'Unknown')}")
    
    # 评估模型性能
    print("\n开始评估模型性能...")
    metrics = evaluate_canet_model(model, test_loader, device, num_classes=3)
    
    # 可视化测试结果
    print("\n开始可视化测试结果...")
    visualize_test_results(model, test_loader, device, num_samples=5)
    
    # 保存评估结果
    results_df = pd.DataFrame({
        'Metric': ['Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Mean', 
                   'IoU_Class0', 'IoU_Class1', 'IoU_Class2', 'IoU_Mean',
                   'Contour_Dice', 'Contour_IoU'],
        'Value': [*metrics['dice_per_class'], metrics['dice_mean'],
                  *metrics['iou_per_class'], metrics['iou_mean'],
                  metrics['contour_dice'], metrics['contour_iou']]
    })
    
    results_df.to_csv('canet_model_evaluation_results.csv', index=False)
    print(f"\n评估结果已保存到: canet_model_evaluation_results.csv")
    
    print(f"\n🎉 CA-net模型测试完成！")

if __name__ == '__main__':
    main()
