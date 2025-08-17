import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2

def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    diag = np.sqrt(h*h + w*w)
    dilation = max(1, int(round(dilation_ratio * diag)))
    kernel = np.ones((3,3), np.uint8)
    gt_dilate = cv2.dilate(mask.astype('uint8'), kernel, iterations=dilation)
    gt_erode = cv2.erode(mask.astype('uint8'), kernel, iterations=dilation)
    bound = gt_dilate - gt_erode
    return bound.astype('float32')[None]

def mask_to_contour_for_completion(mask, dilation_ratio=0.05):
    # completed contour (thick)
    bound = mask_to_boundary(mask, dilation_ratio=dilation_ratio)
    return bound

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt'])
    return checkpoint.get('epoch', 0)
