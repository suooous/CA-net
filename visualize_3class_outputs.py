import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import CANet
from PIL import Image
import torchvision.transforms as transforms
import os
from dataset import SimpleSegDataset

def load_image(image_path, target_size=(512, 512)):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        # 注意：灰度图不需要ImageNet归一化
    ])
    
    # 转换为3通道以匹配ResNet输入要求
    image_tensor = transform(image)  # [1, H, W]
    image_3ch = image_tensor.repeat(3, 1, 1)  # [3, H, W]
    image_tensor = image_3ch.unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor, image

def load_model(checkpoint_path, device, num_classes=3):
    """加载训练好的模型"""
    model = CANet(pretrained=False, num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"成功加载模型检查点: {checkpoint_path}")
    return model

def visualize_3class_outputs(image_tensor, preds, contour_pred, original_image, output_path):
    """可视化3类分割输出结果"""
    # 转换为numpy数组
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()
    
    # 创建子图布局：3行4列
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 第一行：原始图像和轮廓预测
    # 原始图像
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('原始输入图像', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 轮廓预测热力图
    im1 = axes[0, 1].imshow(contour_np, cmap='hot', alpha=0.8)
    axes[0, 1].set_title('轮廓预测 (CCM输出)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # 轮廓预测叠加在原图上
    axes[0, 2].imshow(original_image, cmap='gray')
    im2 = axes[0, 2].imshow(contour_np, alpha=0.6, cmap='Blues')
    axes[0, 2].set_title('轮廓预测叠加', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # 轮廓预测二值化
    contour_binary = (contour_np > 0.5).astype(np.float32)
    axes[0, 3].imshow(contour_binary, cmap='gray')
    axes[0, 3].set_title('轮廓预测二值化 (>0.5)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # 第二行：多尺度分割预测（类别概率）
    for i, pred in enumerate(preds):
        pred_np = pred.squeeze(0).cpu().numpy()  # [C, H, W]
        
        # 显示类别1的概率（上面的白色区域）
        class1_prob = pred_np[1]  # 类别1的概率
        im = axes[1, i].imshow(class1_prob, cmap='viridis', alpha=0.8)
        axes[1, i].set_title(f'类别1概率 (尺度 {i+1})', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], shrink=0.8)
    
    # 最后一个位置显示类别2的概率
    final_pred = preds[-1].squeeze(0).cpu().numpy()
    class2_prob = final_pred[2]  # 类别2的概率
    im_final = axes[1, 3].imshow(class2_prob, cmap='plasma', alpha=0.8)
    axes[1, 3].set_title('类别2概率 (最终尺度)', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im_final, ax=axes[1, 3], shrink=0.8)
    
    # 第三行：最终分类结果和叠加效果
    # 最终分类结果
    final_class_pred = np.argmax(final_pred, axis=0)  # [H, W]
    im_class = axes[2, 0].imshow(final_class_pred, cmap='tab10', vmin=0, vmax=2)
    axes[2, 0].set_title('最终分类结果', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im_class, ax=axes[2, 0], shrink=0.8, ticks=[0, 1, 2])
    
    # 类别1叠加在原图上
    axes[2, 1].imshow(original_image, cmap='gray')
    class1_mask = (final_class_pred == 1).astype(np.float32)
    im_class1 = axes[2, 1].imshow(class1_mask, alpha=0.6, cmap='Reds')
    axes[2, 1].set_title('类别1叠加 (上面区域)', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    
    # 类别2叠加在原图上
    axes[2, 2].imshow(original_image, cmap='gray')
    class2_mask = (final_class_pred == 2).astype(np.float32)
    im_class2 = axes[2, 2].imshow(class2_mask, alpha=0.6, cmap='Blues')
    axes[2, 2].set_title('类别2叠加 (下面区域)', fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')
    
    # 所有类别叠加
    axes[2, 3].imshow(original_image, cmap='gray')
    # 使用不同颜色显示不同类别
    colored_mask = np.zeros((*final_class_pred.shape, 3))
    colored_mask[final_class_pred == 1] = [1, 0, 0]  # 红色 - 类别1
    colored_mask[final_class_pred == 2] = [0, 0, 1]  # 蓝色 - 类别2
    axes[2, 3].imshow(colored_mask, alpha=0.6)
    axes[2, 3].set_title('所有类别叠加', fontsize=12, fontweight='bold')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存3类输出可视化: {output_path}")

def visualize_contour_completion_comparison(image_tensor, preds, contour_pred, original_image, output_path):
    """可视化轮廓补全前后的对比效果"""
    # 转换为numpy数组
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()
    
    # 创建子图布局：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：原始图像和轮廓分析
    # 原始图像
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('原始输入图像', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # CCM补全后的轮廓预测
    im1 = axes[0, 1].imshow(contour_np, cmap='hot', alpha=0.8)
    axes[0, 1].set_title('CCM补全后的轮廓', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # 轮廓叠加在原图上
    axes[0, 2].imshow(original_image, cmap='gray')
    im2 = axes[0, 2].imshow(contour_np, alpha=0.6, cmap='Blues')
    axes[0, 2].set_title('补全轮廓叠加效果', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # 第二行：轮廓补全质量分析
    # 轮廓二值化结果
    contour_binary = (contour_np > 0.5).astype(np.float32)
    axes[1, 0].imshow(contour_binary, cmap='gray')
    axes[1, 0].set_title('补全轮廓二值化 (>0.5)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 轮廓连续性分析（使用形态学操作检测断点）
    import scipy.ndimage as ndimage
    # 检测轮廓的连通性
    labeled, num_features = ndimage.label(contour_binary)
    axes[1, 1].imshow(labeled, cmap='tab20')
    axes[1, 1].set_title(f'轮廓连通性分析\n(连通区域数: {num_features})', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 轮廓质量评估
    # 计算轮廓的平滑度和连续性
    from scipy import ndimage
    # 使用Sobel算子检测边缘强度
    sobel_x = ndimage.sobel(contour_np, axis=1)
    sobel_y = ndimage.sobel(contour_np, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    axes[1, 2].imshow(edge_magnitude, cmap='viridis', alpha=0.8)
    axes[1, 2].set_title('轮廓边缘强度\n(补全质量指标)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(axes[1, 2].images[0], ax=axes[1, 2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存轮廓补全对比可视化: {output_path}")

def analyze_contour_completion_quality(contour_pred):
    """分析轮廓补全的质量指标"""
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()
    
    print("\n=== CCM轮廓补全质量分析 ===")
    
    # 基本统计
    print(f"轮廓预测统计:")
    print(f"  形状: {contour_np.shape}")
    print(f"  值范围: [{contour_np.min():.4f}, {contour_np.max():.4f}]")
    print(f"  平均值: {contour_np.mean():.4f}")
    print(f"  标准差: {contour_np.std():.4f}")
    
    # 轮廓质量指标
    contour_binary = (contour_np > 0.5).astype(np.float32)
    
    # 连通性分析
    import scipy.ndimage as ndimage
    labeled, num_features = ndimage.label(contour_binary)
    print(f"  二值化后轮廓像素数: {(contour_np > 0.5).sum()}")
    print(f"  连通区域数量: {num_features}")
    
    # 轮廓平滑度分析
    from scipy import ndimage
    sobel_x = ndimage.sobel(contour_np, axis=1)
    sobel_y = ndimage.sobel(contour_np, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    print(f"  平均边缘强度: {edge_magnitude.mean():.4f}")
    print(f"  边缘强度标准差: {edge_magnitude.std():.4f}")
    
    # 轮廓连续性评估
    if num_features == 1:
        print(f"  ✅ 轮廓连续性: 优秀 (单一连通区域)")
    elif num_features <= 3:
        print(f"  ⚠️  轮廓连续性: 良好 ({num_features}个连通区域)")
    else:
        print(f"  ❌ 轮廓连续性: 需要改进 ({num_features}个连通区域)")
    
    # 轮廓完整性评估
    total_pixels = contour_np.size
    contour_pixels = (contour_np > 0.5).sum()
    coverage_ratio = contour_pixels / total_pixels
    
    if coverage_ratio > 0.1:
        print(f"  ✅ 轮廓覆盖率: 充足 ({coverage_ratio:.2%})")
    elif coverage_ratio > 0.05:
        print(f"  ⚠️  轮廓覆盖率: 适中 ({coverage_ratio:.2%})")
    else:
        print(f"  ❌ 轮廓覆盖率: 不足 ({coverage_ratio:.2%})")

def save_3class_outputs(preds, contour_pred, output_dir):
    """保存3类分割的各个输出为单独的图像文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存轮廓预测
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()
    
    # 轮廓预测图
    contour_path = os.path.join(output_dir, 'contour_prediction.png')
    contour_img = (contour_np * 255).astype(np.uint8)
    cv2.imwrite(contour_path, contour_img)
    
    # 轮廓预测热力图
    heatmap_path = os.path.join(output_dir, 'contour_heatmap.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(contour_np, cmap='hot', interpolation='nearest')
    plt.colorbar(label='轮廓概率')
    plt.title('轮廓预测热力图 (CCM输出)')
    plt.axis('off')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存各个尺度的分割预测
    for i, pred in enumerate(preds):
        pred_np = pred.squeeze(0).cpu().numpy()  # [C, H, W]
        
        # 保存每个类别的概率图
        for class_id in range(3):
            class_prob = pred_np[class_id]
            
            # 概率图
            prob_path = os.path.join(output_dir, f'class_{class_id}_prob_scale_{i+1}.png')
            prob_img = (class_prob * 255).astype(np.uint8)
            cv2.imwrite(prob_path, prob_img)
            
            # 热力图
            heatmap_path = os.path.join(output_dir, f'class_{class_id}_heatmap_scale_{i+1}.png')
            plt.figure(figsize=(10, 8))
            plt.imshow(class_prob, cmap='viridis', interpolation='nearest')
            plt.colorbar(label=f'类别{class_id}概率')
            plt.title(f'类别{class_id}概率热力图 (尺度{i+1})')
            plt.axis('off')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # 保存分类结果
        class_pred = np.argmax(pred_np, axis=0)
        class_path = os.path.join(output_dir, f'classification_scale_{i+1}.png')
        class_img = (class_pred * 85).astype(np.uint8)  # 0, 85, 170 for 0, 1, 2
        cv2.imwrite(class_path, class_img)
    
    print(f"保存3类输出到: {output_dir}")

def analyze_3class_outputs(preds, contour_pred):
    """分析3类输出结果的统计信息"""
    print("\n=== 3类输出结果分析 ===")
    
    # 轮廓预测分析
    contour_np = contour_pred.squeeze(0).squeeze(0).cpu().numpy()
    print(f"轮廓预测统计:")
    print(f"  形状: {contour_np.shape}")
    print(f"  值范围: [{contour_np.min():.4f}, {contour_np.max():.4f}]")
    print(f"  平均值: {contour_np.mean():.4f}")
    print(f"  标准差: {contour_np.std():.4f}")
    print(f"  二值化后轮廓像素数: {(contour_np > 0.5).sum()}")
    
    # 分割预测分析
    print(f"\n分割预测统计:")
    for i, pred in enumerate(preds):
        pred_np = pred.squeeze(0).cpu().numpy()  # [C, H, W]
        print(f"  尺度 {i+1}:")
        print(f"    形状: {pred_np.shape}")
        print(f"    值范围: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
        
        # 分析每个类别
        for class_id in range(3):
            class_prob = pred_np[class_id]
            print(f"      类别 {class_id}: 平均值={class_prob.mean():.4f}, 最大值={class_prob.max():.4f}")
        
        # 分类结果统计
        class_pred = np.argmax(pred_np, axis=0)
        for class_id in range(3):
            count = np.sum(class_pred == class_id)
            percentage = (count / class_pred.size) * 100
            print(f"      分类结果 - 类别 {class_id}: {count} 像素 ({percentage:.2f}%)")

def main():
    # 设置参数
    checkpoint_path = 'checkpoints/epoch_1.pth'
    image_path = 'data/train_image/00003.png'
    output_dir = '3class_model_outputs'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(checkpoint_path, device, num_classes=3)
    
    # 加载图像
    print(f"加载图像: {image_path}")
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 模型推理
    print("开始模型推理...")
    with torch.no_grad():
        preds, contour_pred = model(image_tensor)
    
    print("推理完成！")
    print(f"分割预测数量: {len(preds)}")
    print(f"轮廓预测形状: {contour_pred.shape}")
    print(f"分割预测形状: {[p.shape for p in preds]}")
    
    # 分析输出
    analyze_3class_outputs(preds, contour_pred)
    
    # 保存完整可视化
    vis_path = os.path.join(output_dir, '3class_outputs_visualization.png')
    visualize_3class_outputs(image_tensor, preds, contour_pred, original_image, vis_path)
    
    # 保存轮廓补全对比可视化
    contour_completion_vis_path = os.path.join(output_dir, 'contour_completion_comparison.png')
    visualize_contour_completion_comparison(image_tensor, preds, contour_pred, original_image, contour_completion_vis_path)

    # 分析轮廓补全质量
    analyze_contour_completion_quality(contour_pred)
    
    # 保存各个输出
    individual_dir = os.path.join(output_dir, 'individual_outputs')
    save_3class_outputs(preds, contour_pred, individual_dir)
    
    print(f"\n🎉 3类输出可视化完成！")
    print(f"结果保存在: {output_dir}")
    print("\n包含以下文件:")
    print("  - 3class_outputs_visualization.png: 3类输出完整可视化")
    print("  - contour_completion_comparison.png: 轮廓补全前后的对比可视化")
    print("  - individual_outputs/: 各个输出的单独文件")
    print("    - contour_prediction.png: 轮廓预测图")
    print("    - contour_heatmap.png: 轮廓预测热力图")
    print("    - class_*_prob_scale_*.png: 各尺度各类别概率图")
    print("    - class_*_heatmap_scale_*.png: 各尺度各类别热力图")
    print("    - classification_scale_*.png: 各尺度分类结果")

if __name__ == '__main__':
    main()
