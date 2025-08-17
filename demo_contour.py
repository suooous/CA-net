import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import CANet
from PIL import Image
import torchvision.transforms as transforms
import os

def load_image(image_path, target_size=(512, 512)):
    """加载并预处理图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor, image

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = CANet(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"成功加载模型检查点: {checkpoint_path}")
    return model

def visualize_contour_prediction(image_tensor, contour_pred, original_image, output_path):
    """可视化轮廓预测结果"""
    # 转换为numpy数组，确保是2D格式
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()  # 移除batch和channel维度
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    # 轮廓预测热力图
    im1 = axes[1].imshow(contour_np, cmap='hot', alpha=0.8)
    axes[1].set_title('轮廓预测热力图', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # 轮廓预测叠加在原图上
    axes[2].imshow(original_image)
    im2 = axes[2].imshow(contour_np, cmap='Blues', alpha=0.6)
    axes[2].set_title('轮廓预测叠加', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存可视化结果: {output_path}")

def save_contour_map(contour_pred, output_path):
    """保存轮廓预测图为PNG文件"""
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()  # 移除batch和channel维度
    
    # 保存为PNG (0-255范围)
    contour_img = (contour_np * 255).astype(np.uint8)
    cv2.imwrite(output_path, contour_img)
    
    print(f"保存轮廓图: {output_path}")

def main():
    # 设置参数
    checkpoint_path = 'checkpoints/epoch_1.pth'
    image_path = 'data/test_image/t00001.png'  # 使用测试集中的第一张图像
    output_dir = 'contour_demo_results'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        print("请先训练模型或指定正确的检查点路径")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print("请指定正确的图像路径")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(checkpoint_path, device)
    
    # 加载图像
    print(f"加载图像: {image_path}")
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 模型推理
    print("开始模型推理...")
    with torch.no_grad():
        preds, contour_pred = model(image_tensor)
    
    print("推理完成！")
    print(f"轮廓预测形状: {contour_pred.shape}")
    print(f"轮廓预测值范围: [{contour_pred.min():.4f}, {contour_pred.max():.4f}]")
    
    # 保存可视化结果
    vis_path = os.path.join(output_dir, 'contour_visualization.png')
    visualize_contour_prediction(image_tensor, contour_pred, original_image, vis_path)
    
    # 保存轮廓图
    contour_path = os.path.join(output_dir, 'contour_prediction.png')
    save_contour_map(contour_pred, contour_path)
    
    # 保存热力图
    heatmap_path = os.path.join(output_dir, 'contour_heatmap.png')
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()  # 移除batch和channel维度
    plt.figure(figsize=(10, 8))
    plt.imshow(contour_np, cmap='hot', interpolation='nearest')
    plt.colorbar(label='轮廓概率')
    plt.title('轮廓预测热力图')
    plt.axis('off')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n演示完成！结果保存在: {output_dir}")
    print("包含以下文件:")
    print("  - contour_visualization.png: 综合可视化结果")
    print("  - contour_prediction.png: 轮廓预测图")
    print("  - contour_heatmap.png: 轮廓预测热力图")

if __name__ == '__main__':
    main()
