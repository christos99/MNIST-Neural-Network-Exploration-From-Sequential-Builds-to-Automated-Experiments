import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Define the number of features (28*28 pixels for MNIST)
num_features = 784

# Normalize and reshape the data
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

# Define a dictionary to configure the neural network model
model_config = {
    "model_type": Sequential,
    "layers": [
        {"layer_type": Flatten, "input_shape": (num_features,)},
        {"layer_type": Dense, "units": 32, "activation": "relu"},
        {"layer_type": Dense, "units": 10, "activation": "softmax"}
    ]
}

# Function to build the neural network model based on the configuration
def build_model(config):
    model = config["model_type"]()
    for layer_config in config["layers"]:
        layer_type = layer_config["layer_type"]
        layer_params = {key: value for key, value in layer_config.items() if key != "layer_type"}
        model.add(layer_type(**layer_params))
    return model

# Build the model using the defined configuration
model = build_model(model_config)

# Compile the model with specific optimizer, loss function, and metrics
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Output a summary of the model's architecture
model.summary()

# Function to display training and validation performance plots
def plt_show(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Train the model using training data, and validate it using testing data
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Display training and validation performance plots
plt_show(history)
