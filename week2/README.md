# Deep Learning Lab 2 — Neural Network & Hyperparameter Tuning

## Overview

This experiment focuses on building a **fully connected neural network** for image classification using the **Fashion-MNIST dataset**.

A baseline neural network is first developed and evaluated. Hyperparameter optimization is then performed using **Randomized Search with 5-Fold Cross-Validation** to find a better model configuration.

## Dataset

**Fashion-MNIST**

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28
- Number of classes: 10

The images are flattened from 28 × 28 pixels into **784 input features**.

## Experiments Performed

### 1. Baseline Neural Network

The baseline neural network uses the following architecture:

```text
Input (784)
      ↓
Dense (128) + ReLU
      ↓
Dense (64) + ReLU
      ↓
Dense (10) + Softmax
```

The model is trained and evaluated using classification metrics.

### 2. Hyperparameter Optimization

Randomized Search with **5-Fold Cross-Validation** is used to evaluate different combinations of hyperparameters.

The search explores:

- Number of hidden layers
- Number of hidden neurons
- Learning rate
- Optimizer
- Activation function
- Dropout rate
- Batch size
- Number of epochs

A total of **20 random hyperparameter combinations** are evaluated.

### 3. Optimized Model

The best-performing hyperparameter configuration is selected based on cross-validation accuracy and then used to retrain the model on the full training dataset.

## Best Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Hidden Layers | 1 |
| Hidden Neurons | 256 |
| Activation | ReLU |
| Dropout | 0.2 |
| Batch Size | 16 |
| Epochs | 20 |

Best Cross-Validation Accuracy: **84.18%**

## Results

| Metric | Baseline | Optimized |
|---|---:|---:|
| Accuracy | 87.96% | **88.61%** |
| Precision | 87.97% | **88.77%** |
| Recall | 87.96% | **88.61%** |
| F1-Score | 87.86% | **88.35%** |

The optimized model improved test accuracy from **87.96% to 88.61%**.

## Key Observations

- Hyperparameter tuning produced a small improvement in model performance.
- The optimized model achieved better accuracy, precision, recall, and F1-score.
- The optimized configuration required more training time than the baseline model.
- Some visually similar Fashion-MNIST classes are more difficult to classify.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Conclusion

This experiment demonstrates the importance of hyperparameter selection in neural networks. Randomized Search and 5-Fold Cross-Validation were used to identify a better configuration, resulting in an improvement over the baseline model.
