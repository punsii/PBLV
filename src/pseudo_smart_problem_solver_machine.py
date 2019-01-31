from src import test_data_reader
from src import generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


# Build TensorFlow model
def build_model(dimension_count, sensor_count):
    model = keras.Sequential([
        keras.layers.Dense(20 * sensor_count, activation=tf.nn.relu, input_shape=(sensor_count,)),
        keras.layers.Dense(20 * sensor_count, activation=tf.nn.softmax),
        keras.layers.Dense(dimension_count)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", "mse"]
    )

    return model


def predict_shit(model, test_distances, test_sensors, test_targets):
    predictions = model.predict(test_distances)

    xAxis = []
    yAxis = []
    for sensor_pos in test_sensors:
        xAxis.append(sensor_pos[0])
        yAxis.append(sensor_pos[1])

    xAxisTargets = []
    yAxisTargets = []
    for i in range(1):
        target = test_targets[i]
        xAxisTargets.append(target[0])
        yAxisTargets.append(target[1])

    xAxisPredictions = []
    yAxisPredictions = []
    for i in range(1):
        prediction = predictions[i]
        xAxisPredictions.append(prediction[0])
        yAxisPredictions.append(prediction[1])

    plt.plot(
        xAxis, yAxis, "ro",
        xAxisTargets, yAxisTargets, "bs",
        xAxisPredictions, yAxisPredictions, "g^"
    )
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.show()


def visualize_error(model, test_sensors, size):
    x = np.arange(size)
    y = np.arange(size)
    z = np.zeros((size, size), dtype=float)

    distances_to_predict = np.zeros(shape=(size * size, len(test_sensors)), dtype=float)
    targets = np.zeros(shape=(size * size, 2), dtype=float)

    for column in x:
        for row in y:
            target = np.array([column, row]) / size
            targets[column * row + row] = target
            distances_to_predict[column * row + row] = generator.calculate_distances(target, test_sensors)

    predictions = model.predict(distances_to_predict)

    for column in x:
        for row in y:
            index = column * row + row

            prediction = predictions[index]
            target = targets[index]

            z[column, row] = np.linalg.norm(prediction - target)

    print("Max error:", z.max(), ", Min error:", z.min(), ", Average error:", z.mean())

    x = np.linspace(0.0, 1.0, size)
    y = np.linspace(0.0, 1.0, size)
    plt.plot()
    plt.contourf(x, y, z)
    plt.show()


# Read data
# sensors, targets, distances = test_data_reader.read_test_data("training", "../")

dimension_count = 2
sensor_count = 3

sensors = generator.generate_targets(sensor_count, dimension_count)
targets, distances = generator.generate_data_matrix(300, dimension_count, sensors)

print(sensors)

print("Dimensions:", dimension_count)
print("Sensors: ", sensor_count)

data_length = len(distances)
data_split = int(data_length * 0.8)
learning_distances, testing_distances = distances[:data_split,:], distances[data_split:,:]
learning_targets, testing_targets = targets[:data_split,:], targets[data_split:,:]

model = build_model(dimension_count, sensor_count)

# Train model
model.fit(learning_distances, learning_targets, epochs=10)

# Test model
test_loss, test_mae, test_mse = model.evaluate(testing_distances, testing_targets)
print("Test MAE:", test_mae, ", Test MSE:", test_mse)

# Plot prediction
predict_shit(model, testing_distances, sensors, testing_targets)

visualize_error(model, sensors, size=100)
