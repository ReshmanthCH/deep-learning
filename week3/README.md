
The dataset, search procedure, best parameters, and final metrics are supported by your notebook. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

# 📙 DL-3 — CNN on CIFAR-10

Rename the notebook to:

`DL-3_CNN_CIFAR10.ipynb`

README:

```markdown
# Deep Learning Lab 3 — CNN for CIFAR-10 Classification

## Overview

This experiment focuses on image classification using a **Convolutional Neural Network (CNN)**.

The model is trained on the **CIFAR-10 dataset** to classify RGB images into 10 different categories.

## Dataset

**CIFAR-10**

- Training images: 50,000
- Testing images: 10,000
- Image dimensions: 32 × 32 × 3
- Number of classes: 10

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## CNN Architecture

The implemented CNN follows this structure:

```text
Input Image
     ↓
Conv2D — 32 Filters
     ↓
MaxPooling
     ↓
Conv2D — 64 Filters
     ↓
MaxPooling
     ↓
Flatten
     ↓
Dense — 128 Neurons
     ↓
Softmax — 10 Classes
