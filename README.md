# MatrixVision

A lightweight forward-pass engine for neural networks, built entirely from scratch using custom matrix operations and raw image preprocessing.

This project loads a `.png` image, converts it to a 28x28 grayscale matrix, flattens it to a vector, and feeds it into a fully connected neural network implemented without any machine learning libraries.

## Features

- Pure Python implementation of a forward-pass neural network
- No TensorFlow, PyTorch, or scikit-learn — all math is manual
- Custom matrix engine: [MatrixLab](https://github.com/yourname/MatrixLab)
- Converts any image to 28×28 grayscale and flattens into 784×1 input vector
- Imports weights for two layers (`input_hidden`, `hidden_output`)
- Performs full forward propagation through 3 layers

## Why this project?

Unlike typical neural networks that rely on high-level libraries, **MatrixVision demonstrates how a neural network works internally**, by implementing:

- Matrix multiplication manually
- Image preprocessing from raw pixels
- Forward pass logic step-by-step

This makes it ideal for:
- Education
- Understanding the mechanics behind neural networks
- Lightweight custom inference

