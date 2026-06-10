# Gender Classification Using Convolutional Neural Networks (CNN)

## Overview

This project develops and evaluates a Convolutional Neural Network (CNN) for binary image classification of male and female faces. The model is trained using TensorFlow and Keras on a publicly available facial image dataset obtained through KaggleHub. Performance is evaluated using accuracy and loss curves, confusion matrices, classification metrics, prediction visualization, and feature map analysis.

## Objectives

- Build a CNN capable of classifying facial images as male or female.
- Evaluate classification performance using standard machine learning metrics.
- Visualize correctly and incorrectly classified images.
- Explore learned image features through convolutional layer activation maps.

## Dataset

**Dataset:** Male and Female Faces Dataset

The dataset is automatically downloaded from Kaggle using KaggleHub:

```python
dataset_path = kagglehub.dataset_download(
    "ashwingupta3012/male-and-female-faces-dataset"
)
```

Dataset structure:

```
Male and Female face dataset/
├── Male/
└── Female/
```

Images are resized to:

- Width: 150 pixels
- Height: 150 pixels

Dataset split:

- Training: 80%
- Validation: 20%

## Software Requirements

### Python Libraries

```python
tensorflow
numpy
matplotlib
scikit-learn
kagglehub
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn kagglehub
```

## Model Architecture

The CNN consists of three convolutional blocks followed by fully connected layers.

### Layer Configuration

| Layer | Parameters |
|---------|------------|
| Conv2D | 100 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 50 filters, 4×4 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 100 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 3×3 |
| Flatten | - |
| Dense | 50 neurons, ReLU |
| Dense | 1 neuron, Sigmoid |

### Training Settings

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross Entropy |
| Batch Size | 32 |
| Epochs | 4 |
| Metric | Accuracy |

Random seeds are fixed to improve reproducibility.

## Data Preprocessing

Images are normalized from the range:

```
0–255
```

to:

```
0–1
```

using:

```python
image / 255.0
```

Dataset caching and prefetching are used to improve training performance.

## Running the Project

Run the Python script:

```bash
python gender_classification.py
```

The script will:

1. Download the dataset.
2. Load and preprocess images.
3. Train the CNN.
4. Plot training accuracy and loss.
5. Generate a confusion matrix.
6. Produce a classification report.
7. Display sample correct and incorrect predictions.
8. Visualize feature maps from convolutional layers.

## Evaluation Metrics

Model performance is assessed using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The classification report is generated using:

```python
classification_report()
```

and the confusion matrix is displayed using:

```python
ConfusionMatrixDisplay()
```

## Visualization Outputs

### Training History

Plots generated:

- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss

### Prediction Examples

The model displays:

- Correctly classified images
- Incorrectly classified images

These examples help identify challenging cases and potential sources of classification error.

### Feature Map Visualization

Intermediate outputs from convolutional and pooling layers are extracted and displayed to illustrate what features the CNN learns during training.

Feature maps are visualized from:

- First convolutional layer
- Third convolutional layer

## Reproducibility

Random seeds are set for:

```python
random
numpy
tensorflow
```

using:

```python
SEED = 42
```

to improve experiment reproducibility.

## Results

The model outputs:

- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix
- Classification report
- Correct prediction examples
- Incorrect prediction examples
- CNN feature map visualizations

These outputs provide insight into both predictive performance and learned image representations.

## Author

Calef Alesse

Chapman University

Data Science / Machine Learning Project
