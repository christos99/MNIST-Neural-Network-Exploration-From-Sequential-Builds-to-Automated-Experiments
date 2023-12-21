# MNIST Neural Network Experiments

## Overview

The project's aim is to provide an educational and practical insight into how different neural network architectures and parameters impact the model's performance, particularly in recognizing handwritten digits from the MNIST dataset.

## Files

### File 1: Initial Experimentation Script

- **Filename**: model_builder_seq.py
- **Purpose**: This script serves as the foundation for experimenting with neural networks on the MNIST dataset. It sequentially defines, compiles, trains, and evaluates multiple neural network models with varying architectures.
- **Key Features**:
  - Loads and preprocesses the MNIST data.
  - Defines multiple neural network models with different configurations (e.g., varying numbers of neurons and layers).
  - Each model is trained and evaluated separately, with its performance visualized using the `plt_show` function.

### File 2: Configurable Model Script

- **Filename**: model_builder_loop.py
- **Purpose**: Enhances the modular approach to configuring and testing neural network models. It focuses on a single, configurable model setup, using a dictionary to define the model's architecture.
- **Key Features**:
  - Introduces a configuration dictionary for flexible model definition.
  - Includes a `build_model` function to construct the model based on the given configuration.
  - The model is compiled, trained, and its performance is visualized with a plot of training and validation accuracy.

### File 3: Automated Neural Network Experiments Script

- **Filename**: mnist_nn_experiments.py
- **Purpose**: This script is an advanced version that enables automated and systematic experimentation with multiple neural network configurations.
- **Key Features**:
  - Functions for creating and training models (`create_model`) and running experiments (`run_experiment`) with specified configurations.
  - Predefined list of experiments with various layer setups and optimizers (SGD, Adam) for easy comparison.
  - Automated execution of experiments and visualization of results, allowing for a comprehensive comparison of different architectures and parameters.

## Usage

1. Ensure Python and necessary libraries (TensorFlow, NumPy, Matplotlib) are installed.
2. Start with `model_builder_loop.py` for a single model demonstration.
3. Progress to `mnist_nn_experiments.py` to execute multiple experiments with varied configurations.

## Conclusion

This project effectively demonstrates the progression from manual, sequential model training to a more structured, modular approach, and finally to an automated, scalable experimentation framework. It's an excellent resource for understanding the impact of neural network configurations on digit recognition tasks and provides a solid foundation for further exploration and learning in the field of neural networks and machine learning.
