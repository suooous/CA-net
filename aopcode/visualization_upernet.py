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

# %%
data_root = r'./data'
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
# make an output folder
out_dir = r"predictions"
os.makedirs(out_dir, exist_ok=True)

# turn off gradients
with torch.no_grad():
    for idx, (imgs, lbls) in enumerate(test_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)['fuse']                       # (1,3,H,W)
        pred_labels = torch.argmax(preds, dim=1)  # (1,H,W)

        # move to cpu numpy
        img_np   = imgs.cpu().squeeze().numpy()       # (H,W)
        lbl_np   = lbls.cpu().squeeze().numpy().astype(np.uint8)
        pred_np  = pred_labels.cpu().squeeze().numpy().astype(np.uint8)

        H, W = img_np.shape

        # build color masks for GT and pred
        def make_color_mask(mask):
            # mask: H×W, values 0/1/2
            cm = np.zeros((H, W, 3), dtype=np.uint8)
            cm[mask == 1] = [255, 0, 0]   # red
            cm[mask == 2] = [0, 255, 0]   # green
            return cm

        lbl_cm  = make_color_mask(lbl_np)
        pred_cm = make_color_mask(pred_np)

        # plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title("Input")
        axes[1].imshow(lbl_cm)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_cm)
        axes[2].set_title("Prediction")

        for ax in axes:
            ax.axis('off')

        # save
        out_path = os.path.join(out_dir, f"sample_{idx:03d}.jpg")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

# %%