
import torch
import torch.nn.functional as F

def dice_score(preds: torch.Tensor,
               targets: torch.Tensor,
               num_classes: int = 3,
               eps: float = 1e-6
              ):
    """
    Compute per-class and mean Dice coefficient.

    Args:
        preds: logits or probabilities, shape (N, C, H, W)
        targets: ground-truth labels, shape (N, H, W), ints in [0..C-1]
        num_classes: number of classes C
        eps: smoothing to avoid zero division

    Returns:
        dice_per_class: Tensor[C] of Dice for each class
        dice_mean:      scalar, mean(Dice over classes)
    """
    # convert to hard labels
    pred_labels = torch.argmax(preds, dim=1)     # (N, H, W)
    dice_per_class = []
    for cls in range(num_classes):
        pred_mask   = (pred_labels == cls).float()  # (N, H, W)
        target_mask = (targets    == cls).float()
        intersection = (pred_mask * target_mask).sum()
        denom        = pred_mask.sum() + target_mask.sum()
        dice_cls     = (2. * intersection + eps) / (denom + eps)
        dice_per_class.append(dice_cls)
    dice_per_class = torch.stack(dice_per_class)   # (C,)
    dice_mean      = dice_per_class.mean()
    return dice_per_class, dice_mean


def iou_score(preds: torch.Tensor,
              targets: torch.Tensor,
              num_classes: int = 3,
              eps: float = 1e-6
             ):
    """
    Compute per-class and mean Intersection-over-Union (IoU).

    Args:
        preds: logits or probabilities, shape (N, C, H, W)
        targets: ground-truth labels, shape (N, H, W), ints in [0..C-1]
        num_classes: number of classes C
        eps: smoothing to avoid zero division

    Returns:
        iou_per_class: Tensor[C] of IoU for each class
        iou_mean:      scalar, mean(IoU over classes)
    """
    pred_labels = torch.argmax(preds, dim=1)       # (N, H, W)
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask   = (pred_labels == cls).float()
        target_mask = (targets    == cls).float()
        intersection = (pred_mask * target_mask).sum()
        union        = pred_mask.sum() + target_mask.sum() - intersection
        iou_cls      = (intersection + eps) / (union + eps)
        iou_per_class.append(iou_cls)
    iou_per_class = torch.stack(iou_per_class)     # (C,)
    iou_mean      = iou_per_class.mean()
    return iou_per_class, iou_mean


