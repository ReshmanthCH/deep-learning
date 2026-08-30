# Deep Learning Laboratory - Experiment 5

## Image Classification using Transfer Learning

This experiment studies image classification using a pretrained MobileNetV2 model on the Oxford-IIIT Pet Dataset.

The main idea of the experiment is to start with a model that has already learned useful visual features from ImageNet and adapt it for classifying different pet breeds. Along with the basic transfer learning model, several experiments are performed to understand how different training choices affect the final performance.

---

## Dataset

The experiment uses the **Oxford-IIIT Pet Dataset**.

The dataset contains images belonging to 37 different pet categories.

### Dataset Details

- Dataset: Oxford-IIIT Pet Dataset
- Number of classes: 37
- Image type: RGB
- Input image size: 224 × 224 × 3
- Official train/validation images: 3680
- Official test images: 3669
- Training samples used: 2944
- Validation samples used: 736
- Test samples used: 3669
- Batch size: 32
- Random seed: 42

The official training portion is shuffled and divided into 80% training and 20% validation data. The official test set is kept separate and is not used during training.

The notebook downloads the dataset automatically, so the dataset does not have to be uploaded manually.

---

## Model Used

The main model used in this experiment is **MobileNetV2** with pretrained ImageNet weights.

The pretrained MobileNetV2 feature extractor is initially kept frozen. A small classification head is added on top of it.

The basic model structure is:

```text
Input Image
    ↓
MobileNetV2
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Layer (37 classes)
    ↓
Softmax Output

The input image is resized to 224 × 224 pixels and MobileNetV2 preprocessing is applied before the image is passed to the network.

The model contains:

MobileNetV2 feature extractor
GlobalAveragePooling2D
Dropout with rate 0.2
Dense output layer with 37 units
Softmax activation

The pretrained backbone contains 2,257,984 non-trainable parameters, while the classification layer contains 47,397 trainable parameters.

Experiments Performed

The notebook includes several experiments instead of relying on only one training configuration. The purpose is to observe how different choices affect the training process and validation performance.

1. Weight Initialization

Different initialization strategies are compared:

Zero initialization
Random initialization
Xavier / Glorot initialization
He initialization

Training loss and validation accuracy are recorded for each approach.

2. Regularization

Regularization techniques are studied to reduce overfitting and improve generalization.

Training and validation accuracy/loss are compared to understand how the model behaves on training data and unseen validation data.

3. Batch Normalization

The effect of batch normalization on model training is investigated.

The experiment compares the training behaviour with and without batch normalization and observes the resulting accuracy and loss curves.

4. Optimizer Comparison

Different optimizers are tested using the same classification setup.

The experiment compares:

SGD
Adam
RMSprop

Their training loss and validation accuracy are plotted so that their behaviour can be compared.

5. Learning Rate Tuning

Different learning-rate values are tested to observe how the step size affects training.

The resulting validation performance is used to identify a suitable learning rate for the experiment.

6. Batch Size Tuning

Different batch sizes are tested and their effect on validation performance is observed.

This helps show how batch size can affect both the training process and the final model performance.

7. Dropout Tuning

Different dropout values are compared.

The purpose is to see how much regularization is useful without making the model unnecessarily difficult to train.

8. Feature Extraction vs Fine-Tuning

Two transfer-learning approaches are compared.

Feature Extraction

The pretrained MobileNetV2 layers are frozen and only the new classification layer is trained.

Fine-Tuning

Selected pretrained layers are allowed to train along with the new classification head.

Their performance and training behaviour are compared.

9. Transfer Learning

The pretrained MobileNetV2 model is adapted to the 37-class pet classification problem.

Training and validation loss are plotted to observe how the model improves during training.

10. Five-Fold Cross-Validation

Five-fold cross-validation is performed to get a more reliable estimate of model performance.

The dataset is divided into five parts. Each part is used as the validation set once, while the remaining four parts are used for training.

The mean accuracy and variation across the folds are reported.

11. Final Test Evaluation

The final model is evaluated on the untouched test set.

The evaluation includes:

Accuracy
Precision
Recall
F1-score
Confusion matrix
Misclassified images
Results

The final independent test evaluation obtained approximately:

Metric	Result
Accuracy	90.57%
Weighted Precision	90.74%
Weighted Recall	90.57%
Weighted F1-score	90.52%

The five-fold cross-validation experiment produced an average accuracy of approximately:

90.30% ± 1.30%

These values are based on the recorded experiment run in the notebook.

Project Structure

A simple project structure can be maintained as follows:

deep-learning/
│
├── lab5.ipynb
├── README.md
├── requirements.txt
│
└── images/
    ├── initialization_training_loss.png
    ├── initialization_validation_accuracy.png
    ├── regularization_training_accuracy.png
    ├── regularization_validation_accuracy.png
    ├── regularization_loss_comparison.png
    ├── batch_normalization_comparison.png
    ├── optimizer_training_loss.png
    ├── optimizer_validation_accuracy.png
    ├── learning_rate_tuning.png
    ├── batch_size_tuning.png
    ├── dropout_tuning.png
    ├── feature_extraction_vs_finetuning.png
    ├── transfer_learning_loss.png
    ├── five_fold_cross_validation.png
    ├── confusion_matrix.png
    └── misclassified_images.png
Running the Experiment
Google Colab

The notebook was designed to work with Google Colab.

Open lab5.ipynb in Google Colab.
Select a suitable runtime, preferably with GPU support.
Run the cells in order.
The Oxford-IIIT Pet Dataset is downloaded automatically.
The required training, validation and test datasets are created.
Run the experiments and inspect the generated plots.
Local Python Environment

Create a virtual environment if required and install the dependencies:

pip install -r requirements.txt

Then launch Jupyter Notebook:

jupyter notebook

Open lab5.ipynb and run the cells sequentially.

Reproducibility

A random seed of 42 is used in the notebook wherever applicable.

The main dataset and model settings are:

Image Size        : 224 × 224
Batch Size        : 32
Number of Classes : 37
Random Seed       : 42
Base Model        : MobileNetV2
Pretrained Model  : ImageNet
Dropout           : 0.2
Notes

The test set is kept separate from the training process and is used only for the final evaluation.

The experiments are mainly intended to understand the effect of different deep-learning choices rather than to build a production-ready pet classifier.

Since transfer learning is used, the model can make use of visual features learned by MobileNetV2 before being adapted to the pet dataset.

Repository

The complete implementation and related Deep Learning laboratory work are available here:

https://github.com/ReshmanthCH/deep-learning
