from src import test_data_reader

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Build TensorFlow model
def build_model(dimension_count):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(dimension_count)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Split data into targets and distance arrays
def split_data(data):
    targets = []
    distances = []

    for data_set in data:
        targets = data_set[0]
        distances = data_set[1]

    return [targets, distances]


# Read data
data = test_data_reader.readTestData(file_name="../test_data.txt")

dimensionCount = len(data[0][0])
sensorCount = len(data[0][1])

splitted_data = split_data(data)
targets = np.array(splitted_data[0], dtype=float)
distances = np.array(splitted_data[1], dtype=float)

print("Dimensions:", dimensionCount)
print("Sensors: ", sensorCount)

print("Targets:", targets)
print("Distances:", distances)

print("Targets Shape:", targets.shape)
print("Distances Shape:", distances.shape)

model = build_model(dimensionCount)

# Train model
model.fit(distances, targets, epochs=10)
