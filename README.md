# MNIST Neural Network Experiments

This repository contains Python scripts for experimenting with different neural network configurations using the MNIST dataset of handwritten digits.

## Overview

The scripts are designed to facilitate the testing of various neural network architectures and parameters on the MNIST dataset. This allows for a comparative understanding of how different configurations perform in terms of accuracy.

## Files

### 1. `mnist_neural_network.py`

- **Purpose**: This script sets up a basic neural network for the MNIST dataset. It's designed to demonstrate how to load the data, prepare it, define a model, train it, and then evaluate its performance.
- **Features**:
    - Load and normalize MNIST data.
    - Define and compile a neural network model.
    - Train the model and plot training and validation accuracy.

### 2. `mnist_nn_experiments.py`

- **Purpose**: This script is designed to run multiple experiments with different neural network configurations. It allows testing various layer setups and optimizers.
- **Features**:
    - Functions to create and train models with specified configurations.
    - Run experiments with different layer counts and neuron counts.
    - Use different optimizers (SGD, Adam).
    - Plot and compare the performance of different configurations.

## Usage

1. Ensure Python and necessary libraries (TensorFlow, NumPy, Matplotlib) are installed.
2. Run `mnist_neural_network.py` for a single model demonstration.
3. Run `mnist_nn_experiments.py` to execute multiple experiments with various configurations.

## Requirements

- Python
- TensorFlow
- NumPy
- Matplotlib

## Experiment Configurations

The `mnist_nn_experiments.py` script includes several predefined experiments. These experiments vary in the number of layers, number of neurons per layer, and the optimizer used.

## Conclusion

These scripts provide a practical way to understand and compare the performance of different neural network architectures and parameters using the MNIST dataset.

---

Feel free to modify the scripts or add new configurations to further explore the capabilities of neural networks in digit recognition tasks.
