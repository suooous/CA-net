import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import AoPDataset

def visualize_label_mapping():
    """可视化标签映射结果"""
    
    # 数据路径
    data_root = './data'
    train_files = os.listdir(os.path.join(data_root, 'train_image'))
    
    # 选择几个样本进行可视化
    sample_files = train_files[:3]  # 前3个样本
    
    for i, sample_file in enumerate(sample_files):
        # 构建路径
        img_path = os.path.join(data_root, 'train_image', sample_file)
        lbl_path = os.path.join(data_root, 'train_label', sample_file)
        
        # 加载原始图像和标签
        img = Image.open(img_path).convert('L')
        lbl = Image.open(lbl_path)
        
        # 转换为numpy数组
        img_array = np.array(img)
        lbl_array = np.array(lbl)
        
        # 使用数据集类处理标签
        dataset = AoPDataset([img_path], [lbl_path])
        processed_img, processed_lbl = dataset[0]
        
        # 转换回numpy数组用于可视化
        processed_lbl_array = processed_lbl.numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title(f'原始图像: {sample_file}')
        axes[0].axis('off')
        
        # 原始标签
        axes[1].imshow(lbl_array, cmap='gray')
        axes[1].set_title(f'原始标签\n(像素值: {np.unique(lbl_array)})')
        axes[1].axis('off')
        
        # 处理后的标签
        im = axes[2].imshow(processed_lbl_array, cmap='tab10', vmin=0, vmax=2)
        axes[2].set_title(f'处理后标签\n(类别索引: {np.unique(processed_lbl_array)})')
        axes[2].axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[2], ticks=[0, 1, 2])
        cbar.set_label('类别索引')
        
        # 显示统计信息
        print(f"\n=== 样本 {i+1}: {sample_file} ===")
        print(f"原始标签像素值: {np.unique(lbl_array)}")
        print(f"处理后标签类别: {np.unique(processed_lbl_array)}")
        print(f"类别0 (背景) 像素数: {np.sum(processed_lbl_array == 0)}")
        print(f"类别1 (上面白色区域) 像素数: {np.sum(processed_lbl_array == 1)}")
        print(f"类别2 (下面白色区域) 像素数: {np.sum(processed_lbl_array == 2)}")
        
        plt.tight_layout()
        plt.show()

def check_specific_sample(filename):
    """检查特定样本的标签映射"""
    
    data_root = './data'
    img_path = os.path.join(data_root, 'train_image', filename)
    lbl_path = os.path.join(data_root, 'train_label', filename)
    
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        print(f"文件不存在: {filename}")
        return
    
    # 加载原始标签
    lbl = Image.open(lbl_path)
    lbl_array = np.array(lbl)
    
    # 使用数据集类处理
    dataset = AoPDataset([img_path], [lbl_path])
    processed_img, processed_lbl = dataset[0]
    processed_lbl_array = processed_lbl.numpy()
    
    # 显示详细信息
    print(f"\n=== 详细分析: {filename} ===")
    print(f"图像尺寸: {lbl_array.shape}")
    print(f"原始标签像素值: {np.unique(lbl_array)}")
    print(f"处理后标签类别: {np.unique(processed_lbl_array)}")
    
    # 统计每个类别的像素数量
    for i in range(3):
        count = np.sum(processed_lbl_array == i)
        percentage = (count / processed_lbl_array.size) * 100
        print(f"类别 {i}: {count} 像素 ({percentage:.2f}%)")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始标签
    axes[0].imshow(lbl_array, cmap='gray')
    axes[0].set_title(f'原始标签: {filename}')
    axes[0].axis('off')
    
    # 处理后标签
    im = axes[1].imshow(processed_lbl_array, cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title(f'处理后标签')
    axes[1].axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2])
    cbar.set_label('类别索引')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== 标签映射验证工具 ===\n")
    
    # 1. 可视化前几个样本
    print("1. 可视化前3个样本的标签映射...")
    visualize_label_mapping()
    
    # 2. 检查特定样本
    print("\n2. 检查特定样本 (00001.png)...")
    check_specific_sample("00001.png")
    
    print("\n验证完成！")
    print("如果标签映射正确，你应该看到：")
    print("- 类别0 (蓝色): 背景区域")
    print("- 类别1 (橙色): 上面的白色区域")
    print("- 类别2 (绿色): 下面的白色区域")
