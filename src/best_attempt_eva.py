from src import test_data_reader

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Build TensorFlow model
def build_model(dimension_count, sensor_count):
    model = keras.Sequential([
        keras.layers.Dense(20 * sensor_count, activation=tf.nn.relu, input_shape=(sensor_count,)),
        keras.layers.Dense(20 * sensor_count, activation=tf.nn.relu),
        keras.layers.Dense(dimension_count)
    ])

    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=0.001),
        loss="mse",
        metrics=["mae", "mse"]
    )

    return model


# Split data into targets and distance arrays
def split_data(data):
    targets = []
    distances = []

    for data_set in data:
        targets.append(data_set[0])
        distances.append(data_set[1])

    return [targets, distances]


# Read data
data = test_data_reader.read_test_data(file_name="../training_data.txt")

dimensionCount = len(data[0][0])
sensorCount = len(data[0][1])

splitted_data = split_data(data)
targets = np.array(splitted_data[0], dtype=float)
distances = np.array(splitted_data[1], dtype=float)

print("Dimensions:", dimensionCount)
print("Sensors: ", sensorCount)

model = build_model(dimensionCount, sensorCount)

# Train model
model.fit(distances, targets, epochs=10)

# Test model
test_data = test_data_reader.read_test_data(file_name="../test_data.txt")

splitted_data = split_data(data)
test_targets = np.array(splitted_data[0], dtype=float)
test_distances = np.array(splitted_data[1], dtype=float)

test_loss, test_mae, test_mse = model.evaluate(test_distances, test_targets)
print("Test MAE:", test_mae, ", Test MSE:", test_mse)
