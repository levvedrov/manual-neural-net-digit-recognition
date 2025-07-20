# Neural-Net-Engine-Lab: Neural Network from Scratch

A simple neural network for digit classification built **entirely from scratch** in Python — no ML frameworks, no libraries like PyTorch or TensorFlow.  
All matrix operations are implemented via a custom Matrix-Lab (https://github.com/levvedrov/matrix-lab) engine.

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

![IMG_2121 (1)](https://github.com/user-attachments/assets/34b4a7e5-92ab-483c-8d3b-1e2f2181516b)


---

## Architecture

| Layer        | Size    |
|--------------|---------|
| Input        | 784 (28x28 grayscale image) |
| Hidden Layer | 100 neurons |
| Output       | 10 neurons (digit classification: 0-9) |
| Activation   | Sigmoid |


---

## How It Works

**starter.py** → generates random weights and saves them to `/model-1/`  
**train_script.py** → runs training loop for several epochs and plots error graph  
**agent.py** → contains full implementation of forward pass, backpropagation and gradient descent

---
