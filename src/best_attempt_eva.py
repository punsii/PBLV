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

    x = np.linspace(0.0, 1.0, size)
    y = np.linspace(0.0, 1.0, size)
    plt.plot()
    plt.contourf(x, y, z)
    plt.show()


# Read data
sensors, targets, distances = test_data_reader.read_test_data("training", "../")

print(sensors)

dimension_count = len(sensors[0])
sensor_count = len(sensors)

print("Dimensions:", dimension_count)
print("Sensors: ", sensor_count)

model = build_model(dimension_count, sensor_count)

# Train model
model.fit(distances, targets, epochs=10)

# Test model
test_sensors, test_targets, test_distances = test_data_reader.read_test_data("training", "../")

test_loss, test_mae, test_mse = model.evaluate(test_distances, test_targets)
print("Test MAE:", test_mae, ", Test MSE:", test_mse)

# Plot prediction
predict_shit(model, test_distances, test_sensors, test_targets)

visualize_error(model, test_sensors, size=10)
