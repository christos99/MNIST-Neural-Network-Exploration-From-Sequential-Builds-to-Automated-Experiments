import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape data
num_features = 784
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

# Function to create a model based on given configuration
def create_model(layers, optimizer='SGD'):
    model = Sequential()
    model.add(Flatten(input_shape=(num_features,)))
    for units in layers:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to run an experiment
def run_experiment(layers, optimizer, epochs=50, batch_size=32):
    model = create_model(layers, optimizer)
    model.summary()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title(f'Model with layers {layers} - Optimizer: {optimizer}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# List of experiments to run
experiments = [
    {'layers': [32], 'optimizer': 'SGD'},
    {'layers': [64, 32], 'optimizer': 'SGD'},
    {'layers': [128, 64, 32], 'optimizer': 'SGD'},
    {'layers': [32], 'optimizer': 'adam'},
    {'layers': [64, 32], 'optimizer': 'adam'}
]

# Run experiments
for experiment in experiments:
    run_experiment(**experiment)
