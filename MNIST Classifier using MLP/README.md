# MNIST Digit Classifier

A feedforward Artificial Neural Network trained on the MNIST dataset, achieving **97% test accuracy**.

---

## Overview

This project implements a simple ANN from scratch to classify handwritten digits (0–9) using the classic MNIST dataset. No convolutions — just fully connected layers doing the heavy lifting.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **97%** |
| Dataset | MNIST (70,000 images) |
| Model Type | Artificial Neural Network (ANN) |

## Dataset

- **Train set:** 60,000 grayscale images (28×28)
- **Test set:** 10,000 grayscale images (28×28)
- **Classes:** 10 (digits 0–9)

## Model Architecture

```
Input (784)  →  Hidden Layer  →  Hidden Layer  →  Output (10)
```

- Input layer: 784 neurons (flattened 28×28 pixels)
- Hidden layers: Fully connected with ReLU activation
- Output layer: 10 neurons with Softmax activation

## Project Structure

```
mnist-classifier/
├── mnist_classifier.ipynb   # Main notebook
└── README.md
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/mnist-classifier.git
cd mnist-classifier

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
```

## Requirements

```
torch
torchvision
numpy
matplotlib
```

## Tech Stack

- **Framework:** PyTorch
- **Language:** Python 3.x

---

*Part of my ML/Computer Vision portfolio — building toward Traffic Sign Detection (GTSRB) for my FYP.*
