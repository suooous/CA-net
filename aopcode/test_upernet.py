# %%
import os
import numpy as np
from glob import glob
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import AoPDataset
import matplotlib.pyplot as plt
import torch
from loss import DiceCoefficient, DiceLoss
import torch.optim as optim
from upernet import UPerNet
from PIL import Image
import torch.nn.functional as F
from metric import dice_score, iou_score

def evaluate_model(model, loader, device, num_classes=3):
    model.eval()
    dice_sum = torch.zeros(num_classes, device=device)
    iou_sum  = torch.zeros(num_classes, device=device)
    dice_mean_sum = 0.0
    iou_mean_sum  = 0.0
    n = 0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)['fuse']

            dice_pc, dice_m = dice_score(preds, lbls, num_classes)
            iou_pc,  iou_m  = iou_score(preds, lbls, num_classes)

            dice_sum += dice_pc
            iou_sum  += iou_pc
            dice_mean_sum += dice_m
            iou_mean_sum  += iou_m
            n += 1

    dice_per_class = (dice_sum / n).cpu().tolist()
    iou_per_class  = (iou_sum  / n).cpu().tolist()
    dice_mean      = (dice_mean_sum / n).item()
    iou_mean       = (iou_mean_sum  / n).item()

    print(f"Dice per class: {dice_per_class}")
    print(f"Mean  Dice   : {dice_mean:.4f}")
    print(f"IoU  per class: {iou_per_class}")
    print(f"Mean  IoU    : {iou_mean:.4f}")

    return {
        'dice_per_class': dice_per_class,
        'dice_mean': dice_mean,
        'iou_per_class': iou_per_class,
        'iou_mean': iou_mean
    }

# %%
data_root = r'D:\data\dataset'
train_files = os.listdir(os.path.join(data_root, 'train_image'))
test_files = os.listdir(os.path.join(data_root, 'test_image'))

test_img_files = [os.path.join(data_root, 'test_image', test_file) for test_file in test_files]
test_lbl_files = [os.path.join(data_root, 'test_label', test_file) for test_file in test_files]

# %%
test_dataset = AoPDataset(
    img_paths=test_img_files,
    lbl_paths=test_lbl_files,
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UPerNet(in_channel=1, num_class=3).to(device)
model.load_state_dict(torch.load("best_model_upernet.pth"))
model.eval()

# %%
metrics = evaluate_model(model, test_loader, device, num_classes=3)
print(metrics)

# %%