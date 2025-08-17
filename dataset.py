import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from scipy import ndimage

class SimpleSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, num_classes=3):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        imgs = set(os.listdir(img_dir))
        masks = set(os.listdir(mask_dir))
        common = sorted(list(imgs & masks))
        self.ids = common
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def process_label(self, label_array):
        """
        处理标签，将白色区域映射为不同的类别
        0: 背景, 1: 上面的白色区域, 2: 下面的白色区域
        """
        # 找到所有白色像素
        white_pixels = (label_array == 255)
        
        # 使用连通组件分析
        labeled_array, num_features = ndimage.label(white_pixels)
        
        if num_features >= 2:
            # 创建新的标签数组
            new_label = np.zeros_like(label_array)
            
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
            
            # 第一个连通组件（上面的）映射为索引1
            if len(centroids) > 0:
                component1_id = centroids[0][1]
                new_label[labeled_array == component1_id] = 1
            
            # 第二个连通组件（下面的）映射为索引2
            if len(centroids) > 1:
                component2_id = centroids[1][1]
                new_label[labeled_array == component2_id] = 2
                
            return new_label
        else:
            # 如果只有一个连通组件，直接映射为索引1
            new_label = np.zeros_like(label_array)
            new_label[white_pixels] = 1
            return new_label

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # 加载图像并转换为灰度图
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path)
        
        if self.transform:
            sample = self.transform(image=np.array(img), mask=np.array(mask))
            img = sample['image']
            mask = sample['mask']
        else:
            # 转换为numpy数组
            img_array = np.array(img)
            mask_array = np.array(mask)
            
            # 处理标签映射
            processed_mask = self.process_label(mask_array)
            
            # 将1通道转换为3通道以匹配ResNet输入要求
            img_3ch = np.stack([img_array, img_array, img_array], axis=0)  # [3, H, W]
            img = torch.from_numpy(img_3ch).float() / 255.0
            
            # 标签转换为长整型
            mask = torch.from_numpy(processed_mask).long()
        
        return img, mask

class AoPStyleDataset(Dataset):
    """
    完全按照aopcode风格的AOP数据集类
    """
    def __init__(self, img_paths, lbl_paths):
        """
        Args:
            img_paths (list of str): 输入图像路径列表
            lbl_paths (list of str): 标签图像路径列表
        """
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths

    def __len__(self):
        return len(self.img_paths)

    def process_label(self, label_array):
        """
        处理标签，将白色区域映射为不同的类别
        0: 背景, 1: 上面的白色区域, 2: 下面的白色区域
        """
        # 找到所有白色像素
        white_pixels = (label_array == 255)
        
        # 使用连通组件分析
        labeled_array, num_features = ndimage.label(white_pixels)
        
        if num_features >= 2:
            # 创建新的标签数组
            new_label = np.zeros_like(label_array)
            
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
            
            # 第一个连通组件（上面的）映射为索引1
            if len(centroids) > 0:
                component1_id = centroids[0][1]
                new_label[labeled_array == component1_id] = 1
            
            # 第二个连通组件（下面的）映射为索引2
            if len(centroids) > 1:
                component2_id = centroids[1][1]
                new_label[labeled_array == component2_id] = 2
                
            return new_label
        else:
            # 如果只有一个连通组件，直接映射为索引1
            new_label = np.zeros_like(label_array)
            new_label[white_pixels] = 1
            return new_label

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        # 转换为灰度图
        img = img.convert('L')

        # 加载对应的标签
        lbl_path = self.lbl_paths[idx]
        lbl = Image.open(lbl_path)

        # 图像转换为张量并归一化 - 转换为3通道以匹配ResNet
        img_array = np.array(img)
        # 将1通道转换为3通道以匹配ResNet输入要求
        img_3ch = np.stack([img_array, img_array, img_array], axis=0)  # [3, H, W]
        img = torch.from_numpy(img_3ch).float() / 255.0

        # 处理标签
        lbl_array = np.array(lbl)
        processed_lbl = self.process_label(lbl_array)
        lbl = torch.from_numpy(processed_lbl).long()
        
        return img, lbl
