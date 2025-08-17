# CA-net: Contour-Aware Network for Medical Image Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Rethinking boundary detection in deep learning-based medical image segmentation**

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Demo](https://img.shields.io/badge/Demo-Live-orange.svg)](https://github.com/yourusername/ca-net)

</div>

## 📖 Overview

CA-net is a novel **Contour-Aware Network** that addresses the critical challenge of boundary detection in medical image segmentation. Unlike traditional approaches that focus solely on pixel-level classification, CA-net introduces a dedicated **Contour Completion Module (CCM)** and **Contour Injection Module (CIM)** to explicitly learn and refine boundary information.

### 🎯 Key Features

- **🔍 Dual-Task Learning**: Simultaneous segmentation and contour detection
- **🔄 Contour Completion**: Advanced ASPP and self-attention for boundary refinement
- **🎨 Multi-Scale Features**: ResNet-50 backbone with StitchViT integration
- **📊 3-Class Segmentation**: Background, upper region, and lower region classification
- **⚡ Efficient Training**: Optimized loss function combining Dice and contour losses

## 🏗️ Architecture

### Model Components

```
Input Image
    ↓
ResNet-50 Backbone
    ↓
StitchViT Block
    ↓
CCM (Contour Completion Module)
    ├── ASPP (Multi-scale Context)
    ├── Self-Attention (Global Structure)
    └── Contour Prediction
    ↓
CIM (Contour Injection Module)
    ├── Contour Feature Fusion
    └── Enhanced Segmentation
    ↓
Multi-Scale Decoder
    ↓
Final Outputs (Segmentation + Contour)
```

### CCM Module Details

The **Contour Completion Module** integrates:
- **ASPP**: Captures multi-scale contextual information
- **Lite Self-Attention**: Establishes global spatial dependencies
- **U-Net Structure**: Encoder-decoder for feature refinement

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/suooous/CA-net.git
cd ca-net

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Organize your data in the following structure:
```
data/
├── train_image/     # Training images
├── train_label/     # Training labels 
├── test_image/      # Test images
└── test_label/      # Test labels 
```

### Training

```bash
# Start training
python train.py

# Training with custom parameters
python train.py 
```

### Testing & Evaluation

```bash
# Evaluate model performance
python test_cto_model.py

```


### Visual Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/a5c688bc-0510-4764-b7d4-3a090256a3b1" width="800" alt="3-Class Segmentation Results">
  <p><em>3-Class Segmentation Results with Contour Detection</em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/e7a83bdd-8974-4f73-8fc7-c3a4fe9f32f9" width="600" alt="AoP computation">
  <p><em>AoP computation</em></p>
</div>

## 🔧 Model Configuration

### Key Parameters

```python
# Model Configuration
num_classes = 3              # Number of segmentation classes
backbone = 'resnet50'        # Backbone network
ccm_channels = 256          # CCM module channels
cim_channels = 128          # CIM module channels
aspp_dilations = [1,6,12,18] # ASPP dilation rates
```

### Loss Function

The model uses a **multi-task loss function**:

```
Total Loss = Segmentation Loss + α × Contour Loss
           = (0.5 × CE + 0.5 × Dice) + 3.0 × Contour Dice
```

Where:
- **CE**: Cross-Entropy Loss
- **Dice**: Dice Coefficient Loss
- **α = 3.0**: Contour loss weight

## 📁 Project Structure

```
ca-net/
├── model.py                    # Main model implementation (CANet)
├── train.py                    # Training script
├── losses.py                   # Loss functions
├── dataset.py                  # Data loading and preprocessing
├── utils.py                    # Utility functions
├── metric.py                   # Evaluation metrics
├── test_cto_model.py          # Model testing and evaluation
├── visualize_3class_outputs.py # Result visualization
├── demo_contour.py            # Contour detection demo
├── checkpoints/               # Model checkpoints
├── data/                      # Dataset directory
└── requirements.txt           # Dependencies
```

## 🎨 Output Classes

The model produces **3-class segmentation**:

| Class | Color | Description |
|-------|-------|-------------|
| 0     | Black | Background |
| 1     | Red   | Upper region |
| 2     | Blue  | Lower region |

## 🔬 Technical Details

### Feature Fusion Strategy

1. **Multi-Scale Feature Extraction**: ResNet-50 layers (64, 512, 1024, 2048 channels)
2. **StitchViT Integration**: Vision Transformer for global context
3. **CCM Processing**: ASPP + Self-Attention for contour completion
4. **CIM Enhancement**: Contour-guided segmentation refinement

### Training Strategy

- **Optimizer**: Adam with learning rate 1e-4
- **Scheduler**: Cosine annealing
- **Data Augmentation**: Random rotation, flip, scale
- **Batch Size**: 8 (adjustable based on GPU memory)

## 📈 Performance Analysis

### Training Curves

The model typically converges within 50-100 epochs with:
- **Segmentation Loss**: Rapidly decreasing in early epochs
- **Contour Loss**: Gradual improvement with contour refinement
- **Validation Metrics**: Stable improvement across all classes



## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the medical imaging community

</div>
