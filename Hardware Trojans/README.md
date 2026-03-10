*** Download datasets from: https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset ***

Hardware Trojan Detection using Machine Learning

This repository contains implementations of several Machine Learning models for detecting Hardware Trojans (HT) based on side-channel data. 
The project compares different algorithms and evaluates their performance on a Trojan side-channel dataset.

Project Overview

Hardware Trojans are malicious modifications introduced into integrated circuits during the design or manufacturing process. 
Detecting them is difficult using traditional testing techniques.

This project explores Machine Learning approaches for Hardware Trojan detection, using features derived from side-channel analysis.

Implemented models:
- Convolutional Neural Network (CNN)
- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM)
- Random Forest

The goal is to compare their effectiveness in classifying circuits as Trojan-infected or Trojan-free.

Repository Structure
.
├── CNN_HM_model.py                     # CNN model for Hardware Trojan detection
├── MLP_HM_model.py                     # Multi-Layer Perceptron implementation
├── SVM_HM_model.py                     # Support Vector Machine model
├── random_forest_HM_model.py           # Random Forest model
├── Wykresy próbek/                    # Sample plots / visualization of data
├── Machine Learning dla detekcji Hardware Trojans.pdf
│                                      # Project documentation
└── Trojan_side_channel_dataset_document.pdf
                                       # Dataset description
Models
CNN Model

Implemented in:
- CNN_HM_model.py
Uses a Convolutional Neural Network to analyze side-channel features and detect patterns indicating the presence of Hardware Trojans.

MLP Model

Implemented in:
- MLP_HM_model.py
A fully connected neural network used as a baseline deep learning approach.

SVM Model

Implemented in:
- SVM_HM_model.py
Uses Support Vector Machine classification, effective for high-dimensional feature spaces.

Random Forest Model

Implemented in:
- random_forest_HM_model.py
Ensemble-based model using multiple decision trees to improve classification robustness.

Dataset

The dataset used in this project is described in:
- Trojan_side_channel_dataset_document.pdf

It contains side-channel measurements that allow distinguishing between:
- Trojan-free circuits
- Trojan-infected circuits

Requirements

Recommended environment:
- Python 3.8+

Required libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow / keras

Install dependencies:
- pip install numpy pandas matplotlib scikit-learn tensorflow
Running the Models

Example:
- python3 CNN_HM_model.py

or
- python3 SVM_HM_model.py

Each script trains the model and outputs evaluation results.

Documentation

Detailed project description can be found in:
- Machine Learning dla detekcji Hardware Trojans.pdf

It includes:
- theoretical background
- dataset description
- methodology
- results and analysis

Results

The models are evaluated using common classification metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

Performance comparisons help determine which algorithm is most effective for Hardware Trojan detection.

Author

Project created as part of research on Machine Learning for Hardware Trojan Detection. 
- https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset
