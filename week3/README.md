# Deep Learning Lab 3 — CNN for CIFAR-10 Classification

## Overview

This experiment focuses on image classification using a **Convolutional Neural Network (CNN)**.

The CNN is trained on the **CIFAR-10 dataset** to classify RGB images into 10 different categories.

## Dataset

**CIFAR-10**

- Training images: 50,000
- Testing images: 10,000
- Image dimensions: 32 × 32 × 3
- Number of classes: 10

### Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

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
```

## Preprocessing

The input images are normalized before training so that pixel values are represented on a smaller scale.

## Training

The model is trained using:

- Optimizer: Adam
- Epochs: 20
- Batch Size: 64
- Multi-class classification

## Results

After 20 epochs:

| Metric | Result |
|---|---:|
| Training Accuracy | 51.04% |
| Validation Accuracy | 49.74% |
| Validation Loss | 1.4131 |

## Key Observations

- Training accuracy increases during training.
- Training loss decreases, indicating that the model is learning.
- Validation accuracy reaches approximately 50%.
- Training and validation accuracy remain relatively close.
- The model achieves moderate performance on the CIFAR-10 classification task.

## Role of CNN Components

### Convolution

Extracts local visual features such as edges, textures, and patterns from images.

### Max Pooling

Reduces the spatial dimensions of feature maps while retaining important information.

### Flatten

Converts the extracted feature maps into a one-dimensional representation.

### Dense Layer

Uses the extracted features for classification.

### Softmax

Produces probabilities for the 10 output classes.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab

## Conclusion

This experiment demonstrates how Convolutional Neural Networks can automatically learn spatial features from images and use them for classification. The implemented CNN successfully learns from the CIFAR-10 dataset, achieving approximately 50% validation accuracy after 20 epochs.
