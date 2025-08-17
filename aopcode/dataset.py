import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class AoPDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images and their corresponding
    annotation masks (e.g., ellipse masks).
    """
    def __init__(self, img_paths, lbl_paths):
        """
        Args:
            img_paths (list of str): Paths to the input images.
            lbl_paths (list of str): Paths to the label (mask) images.
        """
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        # convert to grayscale (“L” mode = 8-bit pixels, black and white)
        img = img.convert('L')

        # Load the corresponding label (mask)
        lbl_path = self.lbl_paths[idx]
        lbl = Image.open(lbl_path)

        img = torch.from_numpy(np.array(img)).unsqueeze(0) / 255.

        lbl = torch.from_numpy(np.array(lbl)).long()
        
        # 使用连通组件分析来识别两个独立的白色区域
        # 0: 背景, 1: 上面的白色区域, 2: 下面的白色区域
        from scipy import ndimage
        
        lbl_array = np.array(lbl)
        
        # 找到所有白色像素
        white_pixels = (lbl_array == 255)
        
        # 使用连通组件分析
        labeled_array, num_features = ndimage.label(white_pixels)
        
        # print(f"找到 {num_features} 个连通组件")
        
        if num_features >= 2:
            # 创建新的标签数组
            new_lbl = np.zeros_like(lbl_array)
            
            # 获取每个连通组件的质心位置
            centroids = []
            for i in range(1, num_features + 1):
                component_mask = (labeled_array == i)
                if np.any(component_mask):
                    # 计算质心的y坐标（行索引）
                    y_coords, x_coords = np.where(component_mask)
                    centroid_y = np.mean(y_coords)
                    centroids.append((centroid_y, i))
            
            # 根据y坐标排序，y坐标小的在上面
            centroids.sort(key=lambda x: x[0])
            
            # print(f"连通组件质心y坐标: {[c[0] for c in centroids]}")
            
            # 第一个连通组件（上面的）映射为索引1
            if len(centroids) > 0:
                component1_id = centroids[0][1]
                new_lbl[labeled_array == component1_id] = 1
                # print(f"组件 {component1_id} (y={centroids[0][0]:.1f}) -> 类别1")
            
            # 第二个连通组件（下面的）映射为索引2
            if len(centroids) > 1:
                component2_id = centroids[1][1]
                new_lbl[labeled_array == component2_id] = 2
                # print(f"组件 {component2_id} (y={centroids[1][0]:.1f}) -> 类别2")
                
            lbl = torch.from_numpy(new_lbl).long()
        else:
            # 如果只有一个连通组件，直接映射为索引1
            new_lbl = np.zeros_like(lbl_array)
            new_lbl[white_pixels] = 1
            lbl = torch.from_numpy(new_lbl).long()
            # print("只有一个连通组件，映射为类别1")

        # Convert the label to a binary mask
        # Here, threshold at 0.5: adjust if your label has a different scale

        return img, lbl