# Multi-Class Food Image Segmentation using CNNs and Transformers
### End-to-End Semantic Segmentation Pipeline using Deep Learning & Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green?logo=opencv)](https://opencv.org)

---

## Overview

This project implements and compares multiple semantic segmentation architectures on the **FoodSeg103** dataset for pixel-level food image understanding.

The objective is to classify every pixel in a food image into one of **104 semantic classes**, enabling accurate segmentation of complex multi-object food scenes.

The project explores both:

- **CNN-based segmentation architectures**
- **Transformer-based segmentation architectures**

and evaluates their performance using **Mean Intersection over Union (mIoU)**.

---

## Results

| Model | Architecture Type | Test mIoU |
|---|---|---|
| DeepLabV3 (ResNet50) | CNN | 0.2636 |
| DeepLabV3+ (ResNet101) | CNN | 0.2432 |
| **SegFormer** | **Transformer** | **0.3023** |

> SegFormer achieved the best performance with **0.3023 mIoU**, outperforming CNN-based models through transformer-based global attention.

---

## Demo

### DeepLabV3(ResNet50)
![DeepLabV3(ResNet50)](results/DeepLabV3(ResNet50).png)

### DeepLabV3+(ResNet101)
![DeepLabV3+(ResNet101)](results/DeepLabV3plus(ResNet101).png)

### SegFormer
![SegFormer](results/SegFormer.png)

### Models Comparison
![Model Comparison](results/Models_comparison.png)

---

## Pipeline

```mermaid
flowchart TD

    A[ FoodSeg103 Dataset<br/>4983 Train Images<br/>2135 Test Images<br/>104 Classes] --> B

    B[ Step 1 — Data Preprocessing<br/>Resize 640x640<br/>Normalization<br/>Mask Encoding] --> C

    C[ Step 2 — Dataset Split<br/>80% Training<br/>20% Validation] --> D

    D[ Step 3 — Model Development] --> D1 & D2 & D3

    D1[DeepLabV3<br/>ResNet50 Backbone<br/>CNN Baseline] --> E
    D2[DeepLabV3+<br/>ResNet101 Backbone<br/>Improved CNN] --> E
    D3[SegFormer<br/>Transformer Encoder<br/>Global Attention] --> E

    E[ Step 4 — Model Training<br/>CrossEntropy Loss<br/>AdamW Optimizer<br/>Cosine LR Scheduler] --> F

    F[ Step 5 — Validation<br/>Loss Tracking<br/>mIoU Evaluation<br/>Checkpoint Saving] --> G

    G[ Step 6 — Test Evaluation<br/>Quantitative Comparison<br/>Qualitative Visualisation] --> H

    H[ Best Model Selection<br/>SegFormer — 0.3023 mIoU]

    style A  fill:#1F4E79,color:#fff,stroke:#1F4E79
    style B  fill:#2E75B6,color:#fff,stroke:#2E75B6
    style C  fill:#2E75B6,color:#fff,stroke:#2E75B6

    style D  fill:#7D5A00,color:#fff,stroke:#7D5A00
    style D1 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00
    style D2 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00
    style D3 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00

    style E  fill:#155724,color:#fff,stroke:#155724
    style F  fill:#155724,color:#fff,stroke:#155724

    style G  fill:#721C24,color:#fff,stroke:#721C24
    style H  fill:#6F42C1,color:#fff,stroke:#6F42C1
```

---

## Model Architectures

### 1. DeepLabV3 — CNN Baseline

- ResNet50 encoder backbone
- Atrous Spatial Pyramid Pooling (ASPP)
- Multi-scale contextual feature extraction
- Strong baseline for semantic segmentation

---

### 2. DeepLabV3+ — Improved CNN

- ResNet101 backbone
- ASPP module
- Decoder for sharper segmentation boundaries
- Enhanced feature representation capacity

---

### 3. SegFormer — Transformer-Based Model

- MiT Transformer Encoder
- Multi-scale feature extraction
- Lightweight MLP decoder
- Global self-attention mechanism
- Best-performing architecture in this project

---

## SegFormer Architecture

```mermaid
flowchart TD

    A[Input Image<br/>640x640] --> B

    B[Patch Embedding] --> C1

    C1[Transformer Stage 1] --> C2
    C2[Transformer Stage 2] --> C3
    C3[Transformer Stage 3] --> C4
    C4[Transformer Stage 4] --> D

    D[Multi-scale Feature Fusion] --> E

    E[MLP Decoder] --> F

    F[Upsampling] --> G

    G[Final Segmentation Mask<br/>104 Classes]

    style A fill:#1F4E79,color:#fff,stroke:#1F4E79
    style B fill:#2E75B6,color:#fff,stroke:#2E75B6

    style C1 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00
    style C2 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00
    style C3 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00
    style C4 fill:#FFF3CD,color:#7D5A00,stroke:#7D5A00

    style D fill:#155724,color:#fff,stroke:#155724
    style E fill:#155724,color:#fff,stroke:#155724
    style F fill:#155724,color:#fff,stroke:#155724

    style G fill:#6F42C1,color:#fff,stroke:#6F42C1
```

---

## Key Technical Highlights

- Implemented multi-class semantic segmentation across **104 food categories**
- Compared **CNN-based** and **Transformer-based** architectures
- Developed complete PyTorch training & evaluation pipeline
- Used **CrossEntropy Loss**, **AdamW**, and **Cosine Annealing Scheduler**
- Applied transformer-based segmentation using **SegFormer**
- Achieved **0.3023 test mIoU**
- Built qualitative visualisation pipeline for prediction analysis
- Automated checkpoint saving and best model selection

---

## Evaluation Metrics

### CrossEntropy Loss
Used during training to optimise pixel-wise class prediction.

### Mean Intersection over Union (mIoU)
Primary evaluation metric for segmentation quality.

```text
IoU = Intersection / Union
mIoU = Mean IoU across all classes
```

mIoU was preferred over accuracy because it handles class imbalance more effectively.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Image Resolution | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Loss Function | CrossEntropyLoss |
| Evaluation Metric | mIoU |
| GPU | NVIDIA A100 |

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib |
| Environment | Google Colab |
| Hardware | NVIDIA A100 GPU |

---

## Project Structure

```text
food-image-segmentation/
│
├── README.md
│
├── src/
│   └── Food_Image_Segmentation.ipynb
│
├── results/
│   ├── DeeplabV3(ResNet50.png)
│   ├── DeeplabV3+(ResNet101.png)
│   ├── SegFormer.png
│   ├── Model_Comparison.png
│
└── data/
    └── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/food-image-segmentation.git

cd food-image-segmentation

pip install -r requirements.txt
```

---

## Dataset

This project uses the **FoodSeg103** dataset.

Dataset Link:
https://xiongweiwu.github.io/foodseg103.html

Due to dataset size and licensing considerations, raw dataset files are not included in this repository.

After downloading, place the dataset inside:

```text
data/FoodSeg103/
```

---

## Running the Project

Open the notebook inside:

```text
src/food_segmentation.ipynb
```

Train and evaluate models directly in Google Colab or Jupyter Notebook.

---

## Future Improvements

Potential future enhancements include:

- Advanced augmentation strategies
- Mixed precision training
- Larger SegFormer variants
- Class-balanced loss functions
- Real-time inference optimisation
- Per-class IoU analysis

---

## References

```bibtex
@article{wu2021foodseg103,
  title={A Large-Scale Benchmark for Food Image Segmentation},
  author={Wu, Xiongwei and others},
  journal={arXiv preprint arXiv:2105.05409},
  year={2021}
}

@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and others},
  booktitle={NeurIPS},
  year={2021}
}

@article{chen2017deeplab,
  title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets},
  author={Chen, Liang-Chieh and others},
  journal={TPAMI},
  year={2017}
}
```

---

## Author

**AJ**

MSc Computer Science / Artificial Intelligence  
Semantic Segmentation using CNNs and Vision Transformers
