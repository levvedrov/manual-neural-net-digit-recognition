# MNISTPass-sc: Neural Network from Scratch

A simple neural network for digit classification built **entirely from scratch** in Python — no ML frameworks, no libraries like PyTorch or TensorFlow.  
All matrix operations are implemented via a custom `MatrixLab` engine.

This project is designed as an educational example of how neural networks work under the hood.

---

## Features

- Image preprocessing (PNG → matrix)
- Forward pass (sigmoid activation)
- Backpropagation
- Gradient descent optimizer
- Manual save/load of weights
- Training visualization (matplotlib)
- No external ML libraries — **all math coded manually**

---

## Architecture

| Layer        | Size    |
|--------------|---------|
| Input        | 784 (28x28 grayscale image) |
| Hidden Layer | 100 neurons |
| Output       | 10 neurons (digit classification: 0-9) |
| Activation   | Sigmoid |

---
