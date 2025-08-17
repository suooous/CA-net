import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCoefficient(nn.Module):
    """
    计算多类别分割的平均Dice系数
    - preds: float tensor, shape (N, C, H, W), 原始logits或概率
    - targets: long tensor, shape (N, H, W), 每个值在 {0,1,2} 中
    返回:
    - dice: 标量tensor, 在N和C上的平均值
    """
    def __init__(self, num_classes=3, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: (N, C, H, W)
        # targets: (N, H, W)
        # 1) 将logits转换为概率
        probs = F.softmax(preds, dim=1)
        
        # 2) 将targets进行one-hot编码 → (N, C, H, W)
        #    F.one_hot给出 (N, H, W, C) 所以我们需要permute
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes) \
                          .permute(0, 3, 1, 2) \
                          .float()
        
        # 3) 展平空间维度: (N, C, H*W)
        N, C, H, W = probs.shape
        probs_flat   = probs.contiguous().view(N, C, -1)
        target_flat  = targets_onehot.contiguous().view(N, C, -1)
        
        # 4) 计算每个类别的交集和并集
        intersection = (probs_flat * target_flat).sum(-1)         # (N, C)
        cardinality  = probs_flat.sum(-1) + target_flat.sum(-1)  # (N, C)
        
        # 5) 每个批次每个类别的dice分数
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # (N, C)
        
        # 6) 在类别和批次上的平均值
        return dice_score.mean()

class DiceLoss(DiceCoefficient):
    """Dice损失：1 - Dice系数"""
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = super().forward(preds, targets)
        return 1.0 - dice

class DiceLossWithCE(nn.Module):
    """
    结合CrossEntropyLoss和Dice损失:
      loss = (1 - alpha) * CE + alpha * (1 - Dice)
    参数:
        num_classes: 类别数C
        alpha: Dice损失项的权重 (0 = 纯CE, 1 = 纯Dice损失)
        ce_kwargs: 传递给nn.CrossEntropyLoss的参数 (例如 weight, ignore_index)
        smooth: Dice的平滑项
    """
    def __init__(self,
                 num_classes: int = 3,
                 alpha: float = 0.5,
                 smooth: float = 1e-6,
                 **ce_kwargs):
        super().__init__()
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(**ce_kwargs)
        self.dice  = DiceCoefficient(num_classes=num_classes, smooth=smooth)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds: (N, C, H, W) logits
        targets: (N, H, W) long标签，值在 [0..C-1] 中
        """
        ce_loss   = self.ce(preds, targets)
        dice_score = self.dice(preds, targets)
        dice_loss  = 1.0 - dice_score
        return (1 - self.alpha) * ce_loss + self.alpha * dice_loss

# 保持向后兼容的函数
def dice_loss(pred, target, eps=1e-6):
    """二值Dice损失，适用于轮廓预测"""
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    loss = 1.0 - (2*inter + eps) / (union + eps)
    return loss.mean()

def bce_loss(pred, target):
    """二值交叉熵损失，适用于轮廓预测"""
    return F.binary_cross_entropy(pred, target)

def combined_seg_loss(preds, target):
    """组合分割损失，支持多类别 - 现在使用AOP方式"""
    # 使用最后一个预测结果（最终输出）
    final_pred = preds[-1]
    
    # 使用DiceLossWithCE
    criterion = DiceLossWithCE(num_classes=3, alpha=0.5)
    loss = criterion(final_pred, target)
    
    return loss

def contour_completion_loss(pred_contour, gt_contour):
    """轮廓完成损失，保持原有逻辑"""
    l1 = F.l1_loss(pred_contour, gt_contour)
    d = dice_loss(pred_contour, gt_contour)
    return l1 + d

# 新增：专门用于3类分割的损失函数
def multi_class_segmentation_loss(preds, targets, alpha=0.5):
    """
    专门为3类分割设计的损失函数
    preds: 多尺度预测列表，每个元素是 [B, 3, H, W]
    targets: 目标标签 [B, H, W]，值在 {0,1,2} 中
    """
    criterion = DiceLossWithCE(num_classes=3, alpha=alpha)
    
    # 计算所有尺度的损失
    total_loss = 0.0
    for pred in preds:
        total_loss += criterion(pred, targets)
    
    return total_loss / len(preds)
