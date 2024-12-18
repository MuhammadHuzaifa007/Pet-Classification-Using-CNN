# Pet Classification using Convolutional Neural Networks (CNN)

This repository contains a Python project for binary image classification using Convolutional Neural Networks (CNN). The project classifies images as either "Cat" or "Dog" using a pre-trained VGG16 model fine-tuned for this task. Below is a detailed overview of the project, its functionality, and how to use it.

## Introduction

Binary image classification is a common task in computer vision. This project demonstrates the use of transfer learning with the VGG16 model to classify images of cats and dogs. The script automates the entire process from data preparation to model evaluation, making it a comprehensive solution for similar tasks.

## Features

- Reads and processes a dataset of images from a zip file.
- Extracts image features and normalizes the data.
- Implements transfer learning using the pre-trained VGG16 model.
- Fine-tunes the model with additional layers for binary classification.
- Handles class imbalance using class weights.
- Includes data augmentation to improve model generalization.
- Provides visualizations of training and validation accuracy/loss.
- Allows for single image prediction with a graphical display.

## How It Works

### 1. Data Preparation
The script extracts images from a zip file, assigns labels based on file names ("cat" or "dog"), and resizes them to 128x128 pixels. The data is normalized and split into training and validation sets while maintaining class balance.

### 2. Data Augmentation
The training data is augmented using rotation, zoom, shift, shear, and flip transformations to improve the model's robustness to variations in the input data.

### 3. Transfer Learning
The script uses a pre-trained VGG16 model as the base. The top layers of the model are replaced with custom layers, including a fully connected layer, dropout, batch normalization, and a sigmoid activation layer for binary classification.

### 4. Model Training
The model is compiled with a binary cross-entropy loss function and trained using the Adam optimizer. Class weights are computed to address class imbalance. Callbacks such as early stopping and learning rate reduction are used to optimize the training process.

### 5. Evaluation and Visualization
The script plots the training and validation accuracy/loss curves to help analyze the model's performance. Additionally, it allows for predictions on individual images and displays the results.

## Prerequisites

To run this project, ensure you have the following libraries installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- PIL (Pillow)
- tqdm

Install the required libraries using:
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow tqdm
```

## Usage

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Place the zip file containing the dataset in the working directory. The file names should indicate the class (e.g., "cat" or "dog").

3. Run the script:
   ```bash
   python script_name.py
   ```

4. Follow the prompts to provide the zip file path and predict image classes.

## Visualizations

- The script plots the training and validation accuracy/loss graphs to analyze performance.
- For predictions, the script displays the original image alongside the predicted label.

## Output

- The trained model is saved as `best_model.keras`.
- Accuracy and loss graphs are displayed after training.
- Single image predictions can be visualized with their labels.

## Conclusion

This project provides a practical implementation of binary image classification using transfer learning. By leveraging the power of pre-trained models and incorporating techniques like data augmentation and class weighting, it achieves robust performance on the task of cat-dog classification.

Feel free to use this repository as a starting point for similar classification tasks. Contributions and suggestions are welcome!

---

**Authors:** Muhammad Huzaifa Zeb.
             Muhammad Bilal.
