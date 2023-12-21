# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist

# Define a dictionary to configure the neural network model
model_config = {
    "model_type": Sequential,
    "layers": [
        {"layer_type": Flatten, "input_shape": (num_features,)},  # Flatten layer to convert input images into a 1D array
        {"layer_type": Dense, "units": 32, "activation": "relu"},  # First Dense layer with 32 neurons and ReLU activation
        {"layer_type": Dense, "units": 10, "activation": "softmax"}  # Second Dense layer with 10 neurons (output layer) and softmax activation
    ]
}

# Function to build the neural network model based on the configuration
def build_model(config):
    model = config["model_type"]()  # Instantiate the model
    for layer_config in config["layers"]:
        layer_type = layer_config["layer_type"]
        layer_params = {key: value for key, value in layer_config.items() if key != "layer_type"}
        model.add(layer_type(**layer_params))  # Add each layer to the model
    return model

# Build the model using the defined configuration
model = build_model(model_config)

# Compile the model with specific optimizer, loss function, and metrics
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Output a summary of the model's architecture
model.summary()

# Train the model using training data, and validate it using testing data
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Function to display training and validation performance plots (defined in the second file)
plt_show(history)
