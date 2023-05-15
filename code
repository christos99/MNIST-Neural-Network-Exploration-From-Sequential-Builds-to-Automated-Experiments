import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist

def plt_show(history):
    """
    Plots the training and validation accuracy from the history object.

    Args:
        history (keras.callbacks.History): History object containing the training metrics.

    Returns:
        None
    """
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.legend()
        plt.title('Performance on training and validation sets')
        plt.show()
    else:
        print("History object does not contain the required metrics.")

# Prepare MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_features = 784  # Data features - img shape = 28*28

# Data Shape
print("Xtrain Shape:", x_train.shape)
print("Xtest Shape:", x_test.shape)
print("Ytrain Shape:", y_train.shape)
print("Ytest Shape:", y_test.shape)

# Display the first six images of the mnist digit database
fig, axes = plt.subplots(ncols=6, sharex=False, sharey=True, figsize=(15, 6))
for i in range(6):
    axes[i].set_title(y_train[i])
    axes[i].imshow(x_train[i], cmap='inferno')
    axes[i].get_xaxis().set_visible(True)
    axes[i].get_yaxis().set_visible(True)
plt.show()

# Convert DATA into suitable format
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

# Create simple Neural Network model with 128 neurons
model = Sequential()
model.add(Flatten(input_shape=(num_features,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10))

# Compile the Neural Network
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 100 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100)

# Use the plt_show function to display the performance plot
plt_show(history)

# Create simple Neural Network model with 256 neurons
model = Sequential()
model.add(Flatten(input_shape=(num_features,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10))

# Compile the Neural Network
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 50 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)

# Create simple Neural Network model with 256 neurons
model = Sequential()
model.add(Flatten(input_shape=(num_features,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the Neural Network
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 50 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)

# Create simple Neural Network model with 32 neurons
model = Sequential()
model.add(Flatten(input_shape=(num_features,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the Neural Network
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 50 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)

# Fit the train and testing data to the Neural Network for 50 epochs with batch size 256
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)

# Create simple Neural Network model with 128 neurons
model = Sequential()
model.add(Flatten(input_shape=(num_features,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10))

# Compile the Neural Network
model.compile(optimizer='SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 50 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)

# Create simple Neural Network model with 128 neurons
model = Sequential()
model.add(Flatten(input_shape=(num_features,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10))

# Compile the Neural Network
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# See the details of the architecture
model.summary()

# Fit the train and testing data to the Neural Network for 50 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

# Use the plt_show function to display the performance plot
plt_show(history)
