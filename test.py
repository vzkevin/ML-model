import numpy as np
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Create validation set by randomly selecting 20% of training images
val_indices = np.random.choice(
    x_train.shape[0], int(x_train.shape[0] * 0.2), replace=False
)
x_val = x_train[val_indices]
y_val = y_train[val_indices]
x_train = np.delete(x_train, val_indices, axis=0)
y_train = np.delete(y_train, val_indices, axis=0)

# Scale the pixel values to 0-1
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Convert labels to binary class matrices
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Build CNN model
model = Sequential()
model.add(
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=x_train.shape[1:])
)
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
)

# Use ModelCheckpoint to save the best model
checkpoint = ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1
)

# Define the data generator
datagen_train = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen_train.fit(x_train)

# Train the model
history = model.fit(
    datagen_train.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint],
)

# Plot the training and validation accuracy
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
print("Train Loss:", train_loss)
print("Train Accuracy:", train_acc)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)
