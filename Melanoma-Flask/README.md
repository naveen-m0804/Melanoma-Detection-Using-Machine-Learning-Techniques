# Melanoma Detection Using Machine Learning Techniques

This project focuses on detecting melanoma using various deep learning models, including **AlexNet**, **ResNet50**, **VGG16**, and **VGG19**. The model is deployed via a Flask web application for real-time image classification, designed to assist in early detection of melanoma.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Melanoma is one of the deadliest forms of skin cancer, and early detection is critical to improving survival rates. This project applies deep learning to classify dermatoscopic images as either melanoma or benign. Using pre-trained models and data augmentation techniques, we achieve high accuracy in detecting melanoma, offering a valuable tool for dermatology.

## Technologies Used
- **Python**
- **Flask** (for deploying the web application)
- **TensorFlow** and **Keras** (for building and training the models)
- **Albumentations** (for data augmentation)
- **Jupyter Notebooks** (for experimentation and EDA)

## Project Structure
The project contains the following files and directories:
- **`EDA.ipynb`**: Exploratory Data Analysis of the dataset
- **`preaugmentation.ipynb` and `postaugmentation.ipynb`**: Notebooks for data augmentation techniques
- **`pretrained.ipynb`**: Training of models using pre-trained architectures (AlexNet, ResNet50, VGG16, VGG19)
- **`utils.py`**: Utility functions for data processing
- **`app.py`**: Flask application for deployment
- **`final_notebook.ipynb`**: Consolidated notebook with all steps and final model performance
- **`folders.ipynb`, `holdout.ipynb`**: Notebooks for data splitting and validation

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/naveen-m0804/Melanoma-Detection-Using-Machine-Learning-Techniques.git
