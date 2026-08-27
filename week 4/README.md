# Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

## CS3807 – Deep Learning Laboratory | Experiment 4

A comparative study of different Convolutional Neural Network (CNN) architectures on the **CIFAR-10 image classification dataset**, with a focus on CNN evolution, transfer learning, fine-tuning, classification performance, and computational cost.

---

## 📌 Overview

Convolutional Neural Networks have evolved significantly over the years, with different architectures introducing new techniques to improve image classification performance, training efficiency, and the ability to train deeper networks.

This experiment compares the following CNN architectures:

* **LeNet-5**
* **AlexNet**
* **VGG16**
* **GoogLeNet**
* **ResNet18**

Two different approaches are studied:

1. **LeNet-5** is implemented and trained **from scratch** as the baseline.
2. **AlexNet, VGG16, GoogLeNet, and ResNet18** are initialized using **ImageNet pretrained weights** and then fully fine-tuned on CIFAR-10.

The models are compared based on:

* Training accuracy
* Training loss
* Test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Training time
* Accuracy-to-computation trade-off

---

## 🎯 Objectives

The objectives of this experiment are:

* To understand the evolution of CNN architectures.
* To study the architectural differences between LeNet-5, AlexNet, VGG16, GoogLeNet, and ResNet.
* To implement a CNN baseline from scratch.
* To understand and apply transfer learning.
* To fine-tune pretrained CNN architectures on CIFAR-10.
* To compare different CNN architectures using standard classification metrics.
* To analyze the computational cost of different architectures.
* To study the effect of residual and Inception-based architectures.
* To understand dilated convolution and transpose convolution.

---

## 🧠 Learning Outcomes

After completing this experiment, the following concepts are understood:

* Convolutional Neural Networks
* CNN architecture evolution
* LeNet-5
* AlexNet
* VGG16
* GoogLeNet / Inception
* ResNet and residual learning
* Skip connections
* Transfer learning
* Feature extraction
* Full fine-tuning
* ImageNet pretrained models
* CIFAR-10 classification
* Cross-entropy loss
* Adam optimization
* Accuracy, precision, recall and F1-score
* Confusion matrices
* Training-time comparison
* Dilated convolution
* Transpose convolution

---

# 🏗️ CNN Architecture Evolution

The architectures studied in this experiment represent major stages in the development of CNNs.

| Architecture | Year | Major Contribution                                   |
| ------------ | ---: | ---------------------------------------------------- |
| LeNet-5      | 1998 | Early practical CNN architecture                     |
| AlexNet      | 2012 | ReLU, Dropout and GPU-based training                 |
| VGG16        | 2014 | Deep architecture using 3 × 3 convolutions           |
| GoogLeNet    | 2014 | Inception modules and multi-scale feature extraction |
| ResNet       | 2015 | Residual learning and skip connections               |

---

# 1. LeNet-5

LeNet-5 is one of the earliest successful CNN architectures and was originally designed for handwritten digit recognition.

In this experiment, a **LeNet-5 style architecture** is implemented from scratch and used as the baseline model.

### Architecture

```text
Input: 32 × 32 × 3
        │
        ▼
Conv2D (3 → 6, 5 × 5)
        │
       ReLU
        │
   Average Pooling
        │
        ▼
Conv2D (6 → 16, 5 × 5)
        │
       ReLU
        │
   Average Pooling
        │
        ▼
      Flatten
        │
        ▼
Linear (400 → 120)
        │
       ReLU
        │
        ▼
Linear (120 → 84)
        │
       ReLU
        │
        ▼
Linear (84 → 10)
```

### Training

* Initialization: Random
* Input size: 32 × 32
* Batch size: 128
* Epochs: 5
* Optimizer: Adam
* Learning rate: 0.001

### Purpose

LeNet-5 acts as a **from-scratch baseline** to demonstrate the difference between training a small CNN from random initialization and fine-tuning a large pretrained CNN.

---

# 2. AlexNet

AlexNet was a major milestone in the development of deep learning.

Important techniques associated with AlexNet include:

* ReLU activation
* Dropout
* GPU-based training
* Data augmentation
* Deeper CNN architectures

For this experiment, ImageNet pretrained weights are loaded and the final classifier is replaced with a 10-class output layer.

### Classifier Modification

```text
Original classifier:
4096 → 1000

Modified classifier:
4096 → 10
```

### Training

* Pretrained weights: ImageNet
* Input size: 224 × 224
* Batch size: 64
* Epochs: 5
* Optimizer: Adam
* Learning rate: 0.0001

---

# 3. VGG16

VGG16 uses a simple and uniform architecture based mainly on repeated **3 × 3 convolution filters**.

Instead of using large filters, multiple smaller convolution layers are stacked together.

### Key Characteristics

* Deep architecture
* Repeated 3 × 3 convolutions
* Strong feature extraction
* Simple and uniform design
* Large number of parameters
* High computational cost

### Classifier Modification

```text
Original classifier:
4096 → 1000

Modified classifier:
4096 → 10
```

### Training

* Pretrained weights: ImageNet
* Input size: 224 × 224
* Batch size: 64
* Epochs: 3
* Optimizer: Adam
* Learning rate: 0.0001

---

# 4. GoogLeNet

GoogLeNet introduced the **Inception module**.

The Inception architecture performs multiple convolution operations in parallel and combines their outputs.

```text
                    Input
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       1 × 1       3 × 3       5 × 5
        Conv        Conv        Conv
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
                Concatenation
                      │
                      ▼
                    Output
```

The 1 × 1 convolutions also help reduce the number of channels before larger convolution operations.

### Auxiliary Classifiers

GoogLeNet uses two auxiliary classifiers during training.

The total loss is:

```text
Ltotal = Lmain + 0.3 × Laux1 + 0.3 × Laux2
```

The auxiliary classifiers provide additional learning signals during training.

They are not required for the final prediction.

### Classifier Modification

```text
Main classifier:
→ 10 classes

Auxiliary classifier 1:
→ 10 classes

Auxiliary classifier 2:
→ 10 classes
```

### Training

* Pretrained weights: ImageNet
* Input size: 224 × 224
* Batch size: 64
* Epochs: 3
* Optimizer: Adam
* Learning rate: 0.0001

---

# 5. ResNet18

ResNet introduced **residual learning** using skip connections.

Instead of directly learning:

```text
H(x)
```

the residual block learns:

```text
F(x) = H(x) - x
```

and produces:

```text
Output = F(x) + x
```

### Residual Block

```text
              Input x
                 │
          ┌──────┴──────┐
          │             │
          ▼             │
       Conv Layer       │
          │             │
          ▼             │
       Conv Layer       │
          │             │
          └──────┬──────┘
                 │
              Addition
                 │
                 ▼
              Output
```

The skip connection provides a direct path for information and gradients, making deeper networks easier to train.

### Training

* Architecture: ResNet18
* Pretrained weights: ImageNet
* Input size: 224 × 224
* Batch size: 64
* Epochs: 3
* Optimizer: Adam
* Learning rate: 0.0001

---

# 🔄 Transfer Learning

Transfer learning is the process of using knowledge learned by a model on one large dataset and adapting it to another dataset or task.

In this experiment, ImageNet pretrained models are adapted to CIFAR-10.

### Workflow

```text
             ImageNet
                 │
                 ▼
        Pretrained CNN
                 │
                 ▼
       Replace Classifier
                 │
                 ▼
     Resize CIFAR-10 Images
          32 × 32 → 224 × 224
                 │
                 ▼
       ImageNet Normalization
                 │
                 ▼
          Fine-Tuning
                 │
                 ▼
        CIFAR-10 Prediction
```

### Why Transfer Learning?

Pretrained CNNs have already learned useful visual features such as:

* Edges
* Textures
* Shapes
* Object patterns

Therefore, the network does not need to learn all these features from random initialization.

This results in faster convergence and generally better performance.

---

# 🔧 Feature Extraction vs Full Fine-Tuning

Two common transfer learning strategies are:

| Feature Extraction             | Full Fine-Tuning                    |
| ------------------------------ | ----------------------------------- |
| Convolutional base is frozen   | Entire network is trainable         |
| Only new classifier is trained | All layers are updated              |
| Lower computational cost       | Higher computational cost           |
| Faster training                | More expensive training             |
| Usually lower adaptation       | Better adaptation to target dataset |

### Strategy Used

This experiment uses **full fine-tuning** for all pretrained models.

The entire network is trained using a small learning rate:

```text
Learning Rate = 0.0001
```

The smaller learning rate helps preserve useful ImageNet features while allowing the network to adapt to CIFAR-10.

---

# 📦 Dataset

## CIFAR-10

CIFAR-10 is an image classification dataset containing 10 classes of RGB images.

### Dataset Statistics

| Property          |       Value |
| ----------------- | ----------: |
| Training images   |      50,000 |
| Testing images    |      10,000 |
| Number of classes |          10 |
| Image size        | 32 × 32 × 3 |
| Image type        |         RGB |

### Classes

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

# 🖼️ Data Preprocessing

Since LeNet-5 operates on the original CIFAR-10 resolution while ImageNet pretrained models expect larger inputs, two preprocessing pipelines are used.

## Pipeline 1 – LeNet-5

```text
CIFAR-10
32 × 32
   │
   ▼
ToTensor
   │
   ▼
CIFAR-10 Normalization
```

Normalization:

```python
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

---

## Pipeline 2 – Pretrained Models

Used for:

* AlexNet
* VGG16
* GoogLeNet
* ResNet18

```text
CIFAR-10
32 × 32
   │
   ▼
Resize to 224 × 224
   │
   ▼
ToTensor
   │
   ▼
ImageNet Normalization
```

Normalization:

```python
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

The pretrained models use ImageNet normalization because their original weights were learned using this preprocessing.

---

# ⚙️ Experimental Setup

## Hardware

```text
GPU: NVIDIA Tesla T4
Device: CUDA
```

## Software

```text
Python
PyTorch 2.11.0+cu128
Torchvision 0.26.0+cu128
NumPy
Matplotlib
Pandas
Scikit-learn
Seaborn
```

---

# 📋 Hyperparameters

| Model     | Initialization | Input Size | Batch Size | Epochs | Learning Rate |
| --------- | -------------- | ---------: | ---------: | -----: | ------------: |
| LeNet-5   | Random         |    32 × 32 |        128 |      5 |         0.001 |
| AlexNet   | ImageNet       |  224 × 224 |         64 |      5 |        0.0001 |
| VGG16     | ImageNet       |  224 × 224 |         64 |      3 |        0.0001 |
| GoogLeNet | ImageNet       |  224 × 224 |         64 |      3 |        0.0001 |
| ResNet18  | ImageNet       |  224 × 224 |         64 |      3 |        0.0001 |

### Common Settings

```text
Optimizer  : Adam
Loss       : Cross-Entropy Loss
num_workers: 2
pin_memory : True
```

---

# 🧪 Experimental Procedure

The experiment consists of four major tasks.

## Task 1 – Dataset Preparation

1. Load CIFAR-10 using `torchvision.datasets`.
2. Create separate preprocessing pipelines for 32 × 32 and 224 × 224 inputs.
3. Create training and testing datasets.
4. Create data loaders.
5. Display sample CIFAR-10 images.
6. Verify the training and testing dataset sizes.

---

## Task 2 – Baseline CNN

A LeNet-5 style CNN is implemented from scratch.

The model consists of:

```text
Conv2D
↓
ReLU
↓
Average Pooling
↓
Conv2D
↓
ReLU
↓
Average Pooling
↓
Fully Connected Layers
↓
10-Class Output
```

The model is trained for five epochs using:

```text
Optimizer    : Adam
Learning Rate: 0.001
Batch Size   : 128
```

---

## Task 3 – Transfer Learning

For each pretrained architecture:

1. Load ImageNet pretrained weights.
2. Replace the original classifier.
3. Change the output layer to 10 classes.
4. Resize CIFAR-10 images to 224 × 224.
5. Apply ImageNet normalization.
6. Fine-tune the complete network.
7. Record training loss and accuracy.

### Classifier Replacement

| Model     | Modified Layer               |
| --------- | ---------------------------- |
| AlexNet   | `classifier[6]`              |
| VGG16     | `classifier[6]`              |
| GoogLeNet | `fc`, `aux1.fc2`, `aux2.fc2` |
| ResNet18  | `fc`                         |

---

## Task 4 – Model Evaluation

The trained models are evaluated using:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-score
* Classification Report
* Confusion Matrix

---

# 📈 Training Results

## LeNet-5

| Epoch |   Loss | Training Accuracy |
| ----: | -----: | ----------------: |
|     1 | 1.7303 |            36.73% |
|     2 | 1.4251 |            48.69% |
|     3 | 1.3261 |            52.40% |
|     4 | 1.2604 |            54.81% |
|     5 | 1.2056 |            56.98% |

Final training accuracy:

```text
56.98%
```

---

## AlexNet

| Epoch |   Loss | Training Accuracy |
| ----: | -----: | ----------------: |
|     1 | 0.5249 |            81.65% |
|     2 | 0.2832 |            90.20% |
|     3 | 0.1889 |            93.44% |
|     4 | 0.1358 |            95.16% |
|     5 | 0.1019 |            96.45% |

Final training accuracy:

```text
96.45%
```

---

## VGG16

| Epoch |   Loss | Training Accuracy |
| ----: | -----: | ----------------: |
|     1 | 0.4340 |            85.17% |
|     2 | 0.1969 |            93.31% |
|     3 | 0.1215 |            95.79% |

Final training accuracy:

```text
95.79%
```

---

## GoogLeNet

| Epoch |   Loss | Training Accuracy |
| ----: | -----: | ----------------: |
|     1 | 0.9144 |            88.10% |
|     2 | 0.3749 |            96.38% |
|     3 | 0.2410 |            98.34% |

Final training accuracy:

```text
98.34%
```

> Note: GoogLeNet's reported training loss includes the main classifier loss and the two auxiliary classifier losses.

---

## ResNet18

| Epoch |   Loss | Training Accuracy |
| ----: | -----: | ----------------: |
|     1 | 0.3188 |            89.46% |
|     2 | 0.0951 |            96.96% |
|     3 | 0.0417 |            98.74% |

Final training accuracy:

```text
98.74%
```

---

# 🏆 Test Performance

The pretrained models were evaluated on the 10,000-image CIFAR-10 test set.

| Model         |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------- | ---------: | ---------: | ---------: | ---------: |
| AlexNet       |     91.31% |     91.40% |     91.31% |     91.25% |
| VGG16         |     92.51% |     92.65% |     92.51% |     92.50% |
| **GoogLeNet** | **95.08%** | **95.09%** | **95.08%** | **95.07%** |
| ResNet18      |     94.06% |     94.23% |     94.06% |     94.07% |

### Best Test Accuracy

```text
GoogLeNet
95.08%
```

---

# ⏱️ Training Time Comparison

| Model     | Epochs | Total Training Time |
| --------- | -----: | ------------------: |
| LeNet-5   |      5 |              60.7 s |
| AlexNet   |      5 |             460.1 s |
| VGG16     |      3 |            2227.6 s |
| GoogLeNet |      3 |             573.6 s |
| ResNet18  |      3 |             448.8 s |

VGG16 was by far the most computationally expensive model in this experiment.

---

# ⚡ Accuracy vs Computational Cost

| Model     | Test Accuracy | Training Time | Accuracy per 100 s |
| --------- | ------------: | ------------: | -----------------: |
| AlexNet   |        0.9131 |       460.1 s |              0.198 |
| VGG16     |        0.9251 |      2227.6 s |              0.042 |
| GoogLeNet |    **0.9508** |       573.6 s |              0.166 |
| ResNet18  |        0.9406 |       448.8 s |          **0.210** |

### Best Accuracy

**GoogLeNet – 95.08%**

### Best Accuracy/Cost Ratio

**ResNet18 – 94.06% in 448.8 seconds**

---

# 📊 Results Summary

| Model     | Approach          | Test Accuracy | Training Time |
| --------- | ----------------- | ------------: | ------------: |
| LeNet-5   | From Scratch      | Not evaluated |        60.7 s |
| AlexNet   | Transfer Learning |        91.31% |       460.1 s |
| VGG16     | Transfer Learning |        92.51% |      2227.6 s |
| GoogLeNet | Transfer Learning |    **95.08%** |       573.6 s |
| ResNet18  | Transfer Learning |        94.06% |   **448.8 s** |

---

# 📉 Visualizations

The experiment produces the following visualizations:

### 1. Sample CIFAR-10 Images

A grid of sample CIFAR-10 images with their corresponding class labels.

### 2. Training Accuracy vs Epoch

Compares the training accuracy progression of:

* LeNet-5
* AlexNet
* VGG16
* GoogLeNet
* ResNet18

### 3. Training Loss vs Epoch

Shows the reduction in training loss for each architecture.

### 4. Total Training Time

Compares the total training time required by each model.

### 5. Test Performance Metrics

Compares:

* Accuracy
* Precision
* Recall
* F1-score

### 6. ResNet18 Confusion Matrix

Shows the class-wise predictions of ResNet18 on the CIFAR-10 test set.

---

# 🔍 Confusion Matrix Analysis

The ResNet18 confusion matrix is mostly diagonal, indicating that most test images are classified correctly.

The largest confusion occurs between:

```text
Cat → Dog : 126 images
Dog → Cat : 27 images
```

Other smaller confusions occur between visually similar classes such as:

* Ship and airplane
* Automobile and truck
* Bird and deer

The cat/dog confusion is understandable because CIFAR-10 images have a very low resolution of only 32 × 32 pixels.

---

# 📑 ResNet18 Classification Report

| Class      | Precision | Recall | F1-Score |
| ---------- | --------: | -----: | -------: |
| Airplane   |      0.93 |   0.97 |     0.95 |
| Automobile |      0.98 |   0.96 |     0.97 |
| Bird       |      0.95 |   0.92 |     0.93 |
| Cat        |      0.92 |   0.82 |     0.87 |
| Deer       |      0.94 |   0.95 |     0.95 |
| Dog        |      0.83 |   0.94 |     0.88 |
| Frog       |      0.98 |   0.95 |     0.97 |
| Horse      |      0.94 |   0.97 |     0.95 |
| Ship       |      0.97 |   0.95 |     0.96 |
| Truck      |      0.97 |   0.96 |     0.97 |

The weakest class-wise performance is observed for the cat/dog pair.

---

# 🧮 Dilated Convolution

Dilated convolution increases the receptive field without increasing the number of kernel parameters proportionally.

A dilation rate determines the spacing between kernel elements.

### Standard Convolution

```text
D = 1
```

### Dilated Convolution

```text
D > 1
```

Conceptually:

```text
Standard:

[ 1 2 3 ]
[ 4 5 6 ]
[ 7 8 9 ]


Dilated:

[ 1 0 2 ]
[ 0 3 0 ]
[ 4 0 5 ]
```

Zeros represent the gaps introduced between kernel elements.

### Applications

* Semantic segmentation
* Medical image analysis
* Satellite imagery
* Object localization

---

# 🔄 Transpose Convolution

Transpose convolution is used to increase the spatial dimensions of feature maps through learnable upsampling.

Unlike pooling, which reduces spatial resolution, transpose convolution can increase it.

### Applications

* Image super-resolution
* Autoencoders
* GANs
* Image generation
* Semantic segmentation

### Simple Difference

```text
Dilated Convolution
→ Increases receptive field

Transpose Convolution
→ Increases spatial resolution
```

---

# 💡 Key Observations

### 1. Transfer Learning Provides a Strong Advantage

AlexNet achieved **81.65% training accuracy after only one epoch**, while LeNet-5 achieved **36.73%** after its first epoch.

This demonstrates the advantage of starting with pretrained visual features.

---

### 2. GoogLeNet Achieved the Best Test Accuracy

GoogLeNet achieved:

```text
95.08% test accuracy
```

making it the best-performing model according to test accuracy.

---

### 3. ResNet18 Had the Best Accuracy/Time Trade-Off

ResNet18 achieved:

```text
94.06% test accuracy
448.8 seconds training time
```

It provided the best accuracy per 100 seconds among the pretrained models.

---

### 4. VGG16 Was Computationally Expensive

VGG16 required:

```text
2227.6 seconds
```

of total training time.

Despite this high cost, it did not achieve the highest test accuracy.

---

### 5. LeNet-5 Was Extremely Fast but Less Accurate

LeNet-5 required only:

```text
60.7 seconds
```

but achieved only:

```text
56.98% training accuracy
```

This highlights the trade-off between model complexity, training cost and classification performance.

---

### 6. Deeper Does Not Always Mean Better

VGG16 is considerably deeper and computationally expensive, but it did not outperform GoogLeNet or ResNet18 in this experiment.

Architecture design and efficient feature extraction are important in addition to depth.

---

# 🏅 Final Ranking

Based on test accuracy:

```text
1. GoogLeNet  → 95.08%
2. ResNet18   → 94.06%
3. VGG16      → 92.51%
4. AlexNet    → 91.31%
```

Based on accuracy-to-training-time ratio:

```text
1. ResNet18
2. AlexNet
3. GoogLeNet
4. VGG16
```

---

# 🥇 Best Model

## Highest Accuracy

**GoogLeNet**

```text
Test Accuracy: 95.08%
```

---

## Best Accuracy/Cost Balance

**ResNet18**

```text
Test Accuracy: 94.06%
Training Time: 448.8 seconds
Accuracy per 100 seconds: 0.210
```

---

## Fastest Model

**LeNet-5**

```text
Training Time: 60.7 seconds
```

However, it was trained from scratch and its test-set metrics were not recorded in this run.

---

## Most Computationally Expensive

**VGG16**

```text
Training Time: 2227.6 seconds
```

---

# ⚠️ Limitations

The experiment has a few limitations:

1. LeNet-5 was not evaluated on the test set in the recorded run.
2. A separate validation split was not used.
3. Validation accuracy and validation loss were therefore not recorded.
4. The models were trained for different numbers of epochs because of computational cost.
5. The pretrained models use resized 224 × 224 CIFAR-10 images, which increases computational requirements.
6. No extensive data augmentation was applied.
7. GoogLeNet's training loss includes auxiliary classifier losses, so its loss is not directly comparable to the plain training loss of the other models.

---

# 🚀 Possible Extensions

The experiment can be extended by:

* Creating a separate validation split.
* Plotting validation accuracy and validation loss.
* Adding early stopping.
* Evaluating LeNet-5 on the test set.
* Comparing feature extraction against full fine-tuning.
* Comparing Adam with SGD + Momentum.
* Adding `RandomCrop`.
* Adding `RandomHorizontalFlip`.
* Visualizing misclassified images.
* Performing detailed analysis of cat/dog misclassification.
* Experimenting with ResNet50.
* Experimenting with EfficientNetB0.
* Comparing additional modern CNN architectures.

---

# 📁 Project Structure

```text
deep-learning/
│
├── DL_LAB_4.ipynb
│
├── README.md
│
└── data/
    └── CIFAR-10/
```

> The CIFAR-10 dataset is downloaded automatically by the notebook when required and is stored under the `data` directory.

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ReshmanthCH/deep-learning.git
cd deep-learning
```

Install the required Python packages:

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn seaborn
```

For GPU acceleration, install the appropriate PyTorch version for your CUDA environment.

---

# ▶️ Running the Experiment

## Option 1 – Google Colab

1. Open `DL_LAB_4.ipynb`.
2. Upload/open the notebook in Google Colab.
3. Enable GPU acceleration.

In Google Colab:

```text
Runtime
   ↓
Change runtime type
   ↓
Hardware accelerator
   ↓
GPU
```

4. Run the notebook cells sequentially.
5. The CIFAR-10 dataset will be downloaded automatically if required.
6. The models will be trained and evaluated.
7. The required plots and metrics will be generated.

---

## Option 2 – Local Environment

Make sure Python and PyTorch are installed.

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
DL_LAB_4.ipynb
```

and execute the cells sequentially.

A CUDA-enabled GPU is recommended because VGG16 and the other pretrained architectures are computationally expensive.

---

# 📚 Main Python Libraries

| Library      | Purpose                                            |
| ------------ | -------------------------------------------------- |
| PyTorch      | Deep learning framework                            |
| Torchvision  | CNN architectures, pretrained weights and CIFAR-10 |
| NumPy        | Numerical operations                               |
| Matplotlib   | Visualization                                      |
| Pandas       | Result organization                                |
| Scikit-learn | Evaluation metrics                                 |
| Seaborn      | Confusion matrix visualization                     |

---

# 📌 Important Implementation Details

### Device Selection

The notebook automatically selects CUDA when available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, CPU is used.

### Loss Function

```python
nn.CrossEntropyLoss()
```

is used for classification.

### Optimizer

```python
torch.optim.Adam
```

is used for model optimization.

### Output Classes

All models produce:

```text
10 output classes
```

corresponding to the CIFAR-10 classes.

---

# 📊 Experiment Summary

This experiment demonstrates the progression of CNN architectures from simple networks such as LeNet-5 to deeper and more sophisticated architectures such as ResNet.

The most important results are:

```text
LeNet-5
→ From scratch
→ 56.98% training accuracy

AlexNet
→ 91.31% test accuracy

VGG16
→ 92.51% test accuracy

GoogLeNet
→ 95.08% test accuracy
→ Best test accuracy

ResNet18
→ 94.06% test accuracy
→ Best accuracy/cost balance
```

The results demonstrate that pretrained CNNs can achieve significantly better performance than a small CNN trained from scratch, while architectural choices strongly affect both accuracy and computational cost.

---

# 🎓 Conclusion

This experiment compared five important CNN architectures on the CIFAR-10 image classification task.

LeNet-5 was trained from scratch as a baseline, while AlexNet, VGG16, GoogLeNet and ResNet18 used ImageNet pretrained weights followed by full fine-tuning.

Transfer learning provided a significant advantage because the pretrained models already contained useful visual features learned from ImageNet.

Among the pretrained models, **GoogLeNet achieved the highest test accuracy of 95.08%**.

**ResNet18 provided the best balance between accuracy and computational cost**, achieving 94.06% test accuracy in 448.8 seconds.

VGG16 achieved good accuracy but required significantly more training time, while AlexNet provided the lowest test accuracy among the pretrained models.

The experiment also demonstrated the importance of architectural innovations such as:

* ReLU and Dropout in AlexNet
* Small 3 × 3 filters in VGG16
* Inception modules in GoogLeNet
* Residual/skip connections in ResNet

Overall, the experiment shows how CNN architecture design and transfer learning can significantly influence classification accuracy, convergence and computational efficiency.

---

# 📖 References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P.
   **Gradient-Based Learning Applied to Document Recognition**, Proceedings of the IEEE, 1998.

2. Krizhevsky, A., Sutskever, I., & Hinton, G.
   **ImageNet Classification with Deep Convolutional Neural Networks**, NeurIPS, 2012.

3. Simonyan, K., & Zisserman, A.
   **Very Deep Convolutional Networks for Large-Scale Image Recognition**, ICLR, 2015.

4. Szegedy, C. et al.
   **Going Deeper with Convolutions**, CVPR, 2015.

5. He, K., Zhang, X., Ren, S., & Sun, J.
   **Deep Residual Learning for Image Recognition**, CVPR, 2016.

6. Krizhevsky, A.
   **Learning Multiple Layers of Features from Tiny Images**, University of Toronto, 2009.

7. Goodfellow, I., Bengio, Y., & Courville, A.
   **Deep Learning**, MIT Press, 2016.

8. PyTorch Documentation
   https://pytorch.org/docs/

9. Torchvision Documentation
   https://pytorch.org/vision/stable/

---

# 🔗 Repository

Complete source code and notebook:

https://github.com/ReshmanthCH/deep-learning

---

# 👨‍💻 Author

**Reshmanth**

B.Tech Artificial Intelligence & Data Science
Shiv Nadar University Chennai

### Course

**CS3807 – Deep Learning Laboratory**

### Experiment

**Experiment 4 – Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning**
