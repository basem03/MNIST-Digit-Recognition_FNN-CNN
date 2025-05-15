
# Feedforward Neural Network for MNIST Digit Classification

This repository contains a Jupyter notebook (`MNIST-Digit-Recognition_FNN-CNN.ipynb`) implementing a **Feedforward Neural Network (FNN)** and a **Convolutional Neural Network (CNN)** for classifying handwritten digits in the MNIST dataset using PyTorch. The project includes data preprocessing, model training, hyperparameter tuning, and performance evaluation.

## Overview

The notebook demonstrates:
- **Data Preprocessing**: Loads the MNIST dataset, splits it into training (60%), validation (20%), and test (20%) sets using stratified splitting, and applies necessary transformations.
- **Model Implementation**: Defines a simple FNN (`SimpleNN`) with two hidden layers and a CNN with convolutional layers, max-pooling, and dropout for regularization.
- **Training and Evaluation**: Implements custom training and evaluation loops using cross-entropy loss and SGD optimizer.
- **Hyperparameter Tuning**: Experiments with learning rates, batch sizes, and hidden layer sizes to optimize performance.
- **Performance Comparison**: Compares FNN and CNN models based on test accuracy and visualizes results with a confusion matrix.
- **Result Analysis**: Provides insights into model performance and optimal configurations.

## Features

- **Stratified Data Splitting**: Ensures balanced class distribution across training, validation, and test sets.
- **Custom Neural Network**: Implements a configurable FNN with two hidden layers and ReLU activations.
- **CNN Architecture**: Includes convolutional layers, layer normalization, max-pooling, and dropout for robust performance.
- **Hyperparameter Optimization**: Tests various learning rates, batch sizes, and hidden layer sizes.
- **Comprehensive Evaluation**: Reports training/validation loss, accuracy, and a confusion matrix for the test set.
- **Visualization**: Plots training/validation loss and accuracy, and displays a confusion matrix for the CNN model.


## Dataset

The MNIST dataset is automatically downloaded via `torchvision.datasets.MNIST`. It consists of:
- 60,000 training images and 10,000 test images of handwritten digits (0–9).
- Each image is 28x28 pixels, grayscale, with a single channel.
- Labels are integers from 0 to 9.


## Usage

1. **Install Dependencies**:
   - Run the pip command above to install required libraries.
   - Ensure an internet connection for the initial MNIST dataset download.

2. **Run the Notebook**:
   - Open `MNIST-Digit-Recognition_FNN-CNN.ipynb` in Jupyter.
   - Execute cells sequentially to:
     - Load and preprocess the MNIST dataset (stratified split, ToTensor transform).
     - Define and train the FNN (`SimpleNN`) with SGD (learning rate=0.01, 10 epochs, batch size=64).
     - Tune hyperparameters (learning rates: 0.1, 0.01, 0.001, 0.0001; batch sizes: 16, 32, 64, 128; hidden layers: 64-32, 128-64, 256-128, 512-256).
     - Train and evaluate a CNN model with convolutional layers and regularization.
     - Compare FNN and CNN performance and visualize results.

3. **Key Steps**:
   - **Data Preparation**: Stratified splitting ensures balanced classes (60% train, 20% val, 20% test).
   - **FNN Architecture**: Input (784 neurons), two hidden layers (default 128-64), output (10 neurons).
   - **CNN Architecture**: Two convolutional layers (32 and 64 filters), max-pooling, layer normalization, dropout (0.25/0.5), and a fully connected layer (128 neurons).
   - **Training**: Uses cross-entropy loss and SGD optimizer for 10 epochs.
   - **Evaluation**: Reports loss and accuracy for training/validation sets, and test accuracy with a confusion matrix for the CNN.
   - **Hyperparameter Tuning**:
     - **Learning Rate**: 0.01 chosen for stable convergence.
     - **Batch Size**: 64 for balanced speed and performance.
     - **Hidden Layers**: 256-128 for best validation accuracy.
     - **Model Comparison**: CNN outperforms FNN due to better spatial feature extraction.

## Results

- **FNN Performance**:
  - Final validation accuracy: ~91.43% (epoch 10).
  - Train loss decreases from 2.1031 to 0.2852; validation loss from 1.6302 to 0.2991.
- **CNN Performance**:
  - Higher test accuracy compared to FNN (exact value depends on run, typically >95%).
  - Better generalization due to convolutional layers capturing spatial patterns.
- **Key Insights**:
  - Learning rate 0.01 provides the best tradeoff between convergence speed and stability.
  - Batch size 64 balances training speed and generalization.
  - Hidden layer configuration 256-128 achieves optimal FNN performance; larger sizes (512-256) risk overfitting.
  - CNN outperforms FNN due to its ability to learn spatial hierarchies, with dropout and normalization reducing overfitting.
- **Confusion Matrix**: Visualizes CNN test set predictions, highlighting misclassifications (e.g., 8s confused with 3s).

## Notes

- **Overfitting**: Larger hidden layers (512-256) showed signs of overfitting; CNN’s dropout and normalization mitigate this.
- **Computational Efficiency**: Batch size 64 is efficient for CPU execution; larger sizes (128) slightly reduce accuracy.
- **MNIST Simplicity**: The dataset’s simplicity allows high accuracy, but CNNs are more robust for complex image tasks.
- **Reproducibility**: Set `random_state=42` for consistent data splits and results.


For issues or contributions, please contact me basemhesham200318@gmail.com open a pull request or issue on this GitHub repository.
