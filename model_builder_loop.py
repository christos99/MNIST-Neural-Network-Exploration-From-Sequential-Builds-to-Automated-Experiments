import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist

# Define the dictionary to store model configuration
model_config = {
    "model_type": Sequential,
    "layers": [
        {"layer_type": Flatten, "input_shape": (num_features,)},
        {"layer_type": Dense, "units": 32, "activation": "relu"},
        {"layer_type": Dense, "units": 10, "activation": "softmax"}
    ]
}

def build_model(config):
    model = config["model_type"]()
    for layer_config in config["layers"]:
        layer_type = layer_config["layer_type"]
        layer_params = {key: value for key, value in layer_config.items() if key != "layer_type"}
        model.add(layer_type(**layer_params))
    return model

# Example usage:
model = build_model(model_config)

# Compile the Neural Network
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 50 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)
