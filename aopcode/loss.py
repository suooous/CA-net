import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCoefficient(nn.Module):
    """
    Computes mean Dice coefficient for multi-class segmentation.
    - preds: float tensor, shape (N, C, H, W), raw logits or probabilities
    - targets: long tensor, shape (N, H, W), each value in {0,1,2}
    Returns:
    - dice: scalar tensor, average over N and C
    """
    def __init__(self, num_classes=3, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: (N, C, H, W)
        # targets: (N, H, W)
        # 1) turn logits into probabilities
        probs = F.softmax(preds, dim=1)
        
        # 2) one-hot encode targets → (N, C, H, W)
        #    F.one_hot gives (N, H, W, C) so we permute
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes) \
                          .permute(0, 3, 1, 2) \
                          .float()
        
        # 3) flatten spatial dims: (N, C, H*W)
        N, C, H, W = probs.shape
        probs_flat   = probs.contiguous().view(N, C, -1)
        target_flat  = targets_onehot.contiguous().view(N, C, -1)
        
        # 4) compute per-class intersection & union
        intersection = (probs_flat * target_flat).sum(-1)         # (N, C)
        cardinality  = probs_flat.sum(-1) + target_flat.sum(-1)  # (N, C)
        
        # 5) dice score per class per batch
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # (N, C)
        
        # 6) mean over classes and batch
        return dice_score.mean()

        
# --- if you want a loss ---
class DiceLoss(DiceCoefficient):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = super().forward(preds, targets)
        return 1.0 - dice
    

class DiceLossWithCE(nn.Module):
    """
    Combines CrossEntropyLoss with Dice loss:
      loss = (1 - alpha) * CE + alpha * (1 - Dice)
    Args:
        num_classes: number of classes C
        alpha: weight for Dice-loss term (0 = pure CE, 1 = pure Dice-loss)
        ce_kwargs: passed to nn.CrossEntropyLoss (e.g., weight, ignore_index)
        smooth: smoothing term for Dice
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
        targets: (N, H, W) long labels in [0..C-1]
        """
        ce_loss   = self.ce(preds, targets)
        dice_score = self.dice(preds, targets)
        dice_loss  = 1.0 - dice_score
        return (1 - self.alpha) * ce_loss + self.alpha * dice_loss