# Multi-Class Food Image Segmentation using CNNs and Transformers

---

# Objective

Semantic segmentation project implemented using PyTorch on the FoodSeg103 dataset.  
This project compares CNN-based and transformer-based architectures for pixel-level food image segmentation across 104 classes.

---

# Project Overview

The objective of this project is to perform semantic segmentation on complex food images by assigning a class label to every pixel in the image.

Three segmentation architectures were implemented and evaluated:

- DeepLabV3 with ResNet50 (Baseline)
- DeepLabV3+ with ResNet101
- SegFormer (Transformer-based Model)

The project includes:

- Data preprocessing pipeline
- Model training and evaluation
- Quantitative comparison using mIoU
- Qualitative prediction visualisation
- CNN vs Transformer analysis

---

# Dataset

This project uses the FoodSeg103 dataset.

Dataset source:  
https://xiongweiwu.github.io/foodseg103.html

Dataset statistics:

- 4,983 training images
- 2,135 test images
- 104 segmentation classes (including background)

After downloading, place the dataset inside the `data/` directory using the following structure:

```plaintext
data/
└── FoodSeg103/
    ├── Images/
    │   ├── ann_dir/
    │   └── img_dir/
    ├── ImageSets/
    ├── category_id.txt
    └── Readme.txt
