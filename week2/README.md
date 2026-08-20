# Deep Learning Lab 2 — Neural Network & Hyperparameter Tuning

## Overview

This experiment focuses on building a **fully connected neural network** for image classification using the **Fashion-MNIST dataset**.

The experiment first establishes a baseline model and then uses **Randomized Search with 5-Fold Cross-Validation** to identify a better set of hyperparameters.

## Dataset

**Fashion-MNIST**

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28
- Number of classes: 10

The images are flattened from 28 × 28 pixels into **784 input features**.

## Experiments

### 1. Baseline Neural Network

The baseline model uses the following architecture:

```text
Input (784)
     ↓
Dense (128) + ReLU
     ↓
Dense (64) + ReLU
     ↓
Dense (10) + Softmax
