# Deep Learning Lab 1 — Perceptron

## Overview

This experiment focuses on understanding and implementing a **Single Layer Perceptron** for binary classification. The model is implemented from scratch and evaluated on the **Banknote Authentication dataset**.

The experiment also explores learning rates, activation functions, the XOR problem, and the effect of feature normalization.

## Dataset

**Banknote Authentication Dataset**

- Samples: 1372
- Features: 4
  - Variance
  - Skewness
  - Curtosis
  - Entropy
- Classes: 2 (0 and 1)
- Train/Test Split: 80/20

Feature scaling is performed using `StandardScaler`.

## Experiments

### 1. Single Layer Perceptron

A perceptron is implemented from scratch using:
- Weights
- Bias
- Learning rate
- Epochs
- Perceptron weight-update rule

### 2. Learning Rate Comparison

The following learning rates are compared:

- 0.001
- 0.01
- 0.1

The experiment studies how the learning rate affects the training process.

### 3. Step vs Sigmoid Activation

The Step and Sigmoid activation functions are compared based on their behavior and suitability for neural-network learning.

### 4. XOR Problem

The experiment demonstrates why a single-layer perceptron cannot solve the XOR problem.

XOR is not linearly separable, so a single linear decision boundary is insufficient.

### 5. Feature Normalization

Perceptron performance and convergence are compared:

- Without normalization
- With normalization

## Results

The trained perceptron achieved:

| Metric | Score |
|---|---:|
| Accuracy | 98.18% |
| Precision | 96.06% |
| Recall | 100% |
| F1-Score | 97.99% |

## Key Observations

- A perceptron performs well on linearly separable binary classification.
- Learning rate affects the size of weight updates and convergence behavior.
- A single-layer perceptron cannot solve XOR because it is not linearly separable.
- Feature normalization can influence convergence and training stability.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Conclusion

This experiment provides a foundation for understanding how a perceptron learns through weight updates and highlights the limitations of single-layer linear models. It also motivates the use of multilayer neural networks for solving nonlinear problems.
