# Diabetic Retinopathy Detection using Deep Learning

A comprehensive deep learning project for automated classification of diabetic retinopathy stages from retinal fundus images. This research implements and compares multiple CNN architectures, transfer learning approaches, regularization techniques, and ensemble methods to achieve robust medical image classification.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Key Findings](#key-findings)
- [Documentation](#documentation)
- [Citation](#citation)

---

## 🔍 Overview

Diabetic retinopathy (DR) is a leading cause of vision loss among adults with diabetes. This project develops an AI-powered diagnostic system capable of classifying DR severity into 5 stages:

- **Class 0**: No DR (Healthy)
- **Class 1**: Mild Non-Proliferative DR
- **Class 2**: Moderate Non-Proliferative DR
- **Class 3**: Severe Non-Proliferative DR
- **Class 4**: Proliferative DR

### Key Features

✅ **Custom CNN Architecture**: Baseline model built from scratch  
✅ **Transfer Learning**: EfficientNet-B0 & ResNet-50 with fine-tuning strategies  
✅ **Advanced Regularization**: Dropout, Weight Decay, Label Smoothing, Data Augmentation  
✅ **Ensemble Learning**: Soft voting across 5 best models  
✅ **Unsupervised Analysis**: Convolutional Autoencoder for anomaly detection & latent space visualization  
✅ **Comprehensive Evaluation**: ROC-AUC, Confusion Matrix, Learning Curves, t-SNE visualization

---

## 📊 Dataset

**Source**: [Kaggle - Diabetic Retinopathy Detection](https://www.kaggle.com/datasets/jockeroika/diabetic-retinopathy)

- **Total Images**: 2,750
- **Image Size**: 256×256 pixels (RGB)
- **Classes**: 5 (staged severity levels)
- **Split**: 70% train / 15% validation / 15% test

### Class Distribution

| Class | Label | Count |
|-------|-------|-------|
| 0 | Healthy | ~550 |
| 1 | Mild DR | ~450 |
| 2 | Moderate DR | ~600 |
| 3 | Severe DR | ~400 |
| 4 | Proliferative DR | ~500 |

> **Note**: Dataset not included in repository. Download from Kaggle and place in `data/` directory.

---

## 📁 Project Structure
```
DiabeticRetinopathy(CVproj)/
├── README.md
├── requirements.txt
├── checkpoints/                      # Saved model weights
│
├── data/                             # Dataset (not included)
│   ├── Healthy/
│   ├── Mild DR/
│   ├── Moderate DR/
│   ├── Severe DR/
│   └── Proliferate DR/
├── docs/                             # Research documentation (PDF report)
├── exploration/                      # Jupyter notebooks for experiments and visualization
│   ├── baseline.ipynb
│   ├── transfer.ipynb
│   ├── regularization.ipynb
│   └── autoencoder.ipynb
├── models/                           # Model architectures
│   ├── __init__.py
│   ├── baseline_cnn.py
│   ├── transfer_models.py
│   ├── ensemble.py
│   └── autoencoder.py
├── scripts/                          # Training scripts
│   ├── train_baseline.py
│   ├── train_transfer.py
│   ├── train_ensemble.py
│   └── train_autoencoder.py
├── training/                         # Training utilities
│   ├── config.py
│   ├── data_processor.py
│   ├── train_pipeline.py
│   ├── early_stopping.py
│   └── check_gpu.py
└── results/                          # Output metrics
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone repository
git clone https://github.com/deyme17/Diabetic-Retinopathy-CV.git
cd Diabetic-Retinopathy-CV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python training/check_gpu.py
```

### Download Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/jockeroika/diabetic-retinopathy)
2. Extract to `data/` directory maintaining folder structure

---

## 💻 Usage

### Training Models

#### 1. Baseline CNN
```bash
python scripts/train_baseline.py
```

#### 2. Transfer Learning (EfficientNet-B0)
```bash
python scripts/train_transfer.py --model "efficientnet_b0" --freeze_until 2
```

#### 3. Transfer Learning (ResNet-50)
```bash
python scripts/train_transfer.py --model "resnet50" --augmentation 1
```

#### 4. Ensemble Model
```bash
python scripts/train_ensemble.py --models "checkpoint1.pth" "checkpoint2.pth"
```

#### 5. Autoencoder (Anomaly Detection)
```bash
python scripts/train_autoencoder.py --epochs 50
```

### Jupyter Notebooks

Explore experiments interactively:
```bash
jupyter notebook exploration/
```

- `baseline.ipynb` - Custom CNN development
- `transfer.ipynb` - Transfer learning experiments
- `regularization.ipynb` - Dropout, weight decay, augmentation analysis
- `autoencoder.ipynb` - Unsupervised learning & t-SNE visualization

---

## 📈 Models & Results

### Performance Comparison

| Model | Val Accuracy | Test Accuracy | Parameters | Training Time |
|-------|-------------|---------------|------------|---------------|
| **Baseline CNN** | 64.00% | 63.55% | ~3.67M | 21 epochs |
| **EfficientNet-B0** (freeze_until=2) | 72.00% | **69.57%** | ~4.0M | 9 epochs |
| **ResNet-50** (regularized) | 62.00% | 65.66% | ~23.5M | 17 epochs |
| **Ensemble** (Soft Voting) | 70.00% | **70.57%** | Combined | N/A |

### Key Metrics (Best Ensemble Model)

- **Accuracy**: 70.57%
- **Precision**: 0.6537
- **Recall**: 0.6557
- **F1-Score**: 0.6559

### Class-wise Performance (AUC-ROC)

| Class | AUC |
|-------|-----|
| Healthy | 0.99 |
| Mild DR | 0.92 |
| Moderate DR | 0.82 |
| Severe DR | 0.92 |
| Proliferate DR | 0.90 |

### Autoencoder Results

- **Anomalies Detected**: 17/299 test images
- **Detection Threshold**: μ + 2.3σ (MSE-based)
- **Latent Space Dimensionality**: 16×16×256

---

## 🔬 Key Findings

### 1. Transfer Learning Superiority
- **EfficientNet-B0** achieved +6% accuracy over baseline CNN
- Compact architecture (4M params) outperformed ResNet-50 (23.5M params)
- **Partial freezing** (freeze_until=2) optimal for medical imaging domain adaptation

### 2. Effective Regularization
- **Dropout (0.5)** + **Data Augmentation** extended training epochs
- **Weight Decay (1e-4)** reduced validation loss
- **Label Smoothing** proved ineffective (decreased accuracy by 2-3%)

### 3. Ensemble Benefits
- **Soft Voting** increased accuracy by +1% while improving stability
- Critical for **Proliferate DR** detection
- Reduced false negatives in high-risk classes

### 4. Unsupervised Insights
- **t-SNE visualization** revealed continuum between DR stages (no clear cluster boundaries)
- **Healthy class** forms distinct cluster, but **Mild/Moderate DR** overlap significantly
- **Autoencoder** successfully identified low-quality images and severe anomalies

### 5. Challenges
- **Class imbalance** required weighted loss functions
- **Moderate DR** (Class 2) most difficult to classify (often confused with Mild/Severe)
- **Medical domain gap**: ImageNet pretraining requires careful fine-tuning

---

## 📚 Documentation

Full research report available in `docs/DiabeticRetinopathyReport.pdf` (Ukrainian)

### Contents:
1. Problem Statement & Dataset Analysis
2. Theoretical Background (CNNs, Transfer Learning, Autoencoders)
3. Baseline Model Development
4. Transfer Learning Experiments
5. Regularization Techniques
6. Ensemble Methods
7. Autoencoder Analysis
8. Comparative Results & Conclusions