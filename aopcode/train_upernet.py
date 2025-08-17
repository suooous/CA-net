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
from loss import DiceCoefficient, DiceLossWithCE
import torch.optim as optim
from upernet import UPerNet
from PIL import Image
import torch.nn.functional as F
from metric import dice_score, iou_score
# %%
data_root = r'./data'
train_files = os.listdir(os.path.join(data_root, 'train_image'))
test_files = os.listdir(os.path.join(data_root, 'test_image'))

# %%
train_img_files = [os.path.join(data_root, 'train_image', train_file) for train_file in train_files]
train_lbl_files = [os.path.join(data_root, 'train_label', train_file) for train_file in train_files]

test_img_files = [os.path.join(data_root, 'test_image', test_file) for test_file in test_files]
test_lbl_files = [os.path.join(data_root, 'test_label', test_file) for test_file in test_files]


# %%
# Create the datasets
train_dataset = AoPDataset(
    img_paths=train_img_files,
    lbl_paths=train_lbl_files,
)

test_dataset = AoPDataset(
    img_paths=test_img_files,
    lbl_paths=test_lbl_files,
)

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print("DataLoaders for train, validation, and test datasets have been created.")

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UPerNet(in_channel=1, num_class=3).to(device)
# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = DiceLossWithCE()
metric = DiceCoefficient()

# Number of training epochs
num_epochs = 100

best_val_dice = -1.0  # Initialize with a low value

# ------------------------------------------------
# Training and Validation Loop
# ------------------------------------------------
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs)['fuse']
        loss = criterion(preds, lbls)  # dice_loss returns (1 - Dice coefficient)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            preds = model(imgs)['fuse']
            loss = criterion(preds, lbls)
            val_loss += loss.item() * imgs.size(0)
            # Compute dice score (Dice coefficient) for this batch
            batch_dice = metric(preds, lbls)
            val_dice += batch_dice * imgs.size(0)
    
    val_loss /= len(test_loader.dataset)
    val_dice /= len(test_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Dice Score: {val_dice:.4f}")
    
    # Save the model if the validation dice score improved
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_model_upernet.pth")
        print(f"Model saved at epoch {epoch+1} with Validation Dice Score: {val_dice:.4f}")

# %%
