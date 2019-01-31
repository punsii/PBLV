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


def visualize_error(model, sensors, size, dimension_count):
    targets, distances = generator.generate_data_matrix(size, dimension_count, sensors)

    predictions = model.predict(distances)

    def calculate_errors(predicted_targets, original_targets):
        errors = np.zeros((len(predicted_targets),))

        for i in range(len(predicted_targets)):
            errors[i] = np.linalg.norm(predicted_targets[i] - original_targets[i])

        return errors

    def draw_2d_chart(size, errors):
        x = np.arange(size)
        y = np.arange(size)
        z = np.zeros((size, size), dtype=float)

        for column in x:
            for row in y:
                index = column * row + row

                z[column, row] = errors[index]

        x = np.linspace(0.0, 1.0, size)
        y = np.linspace(0.0, 1.0, size)
        plt.plot()
        plt.contourf(x, y, z)
        plt.show()

    def draw_3d_chart(size, errors):
        x = np.arange(size)
        y = np.arange(size)
        z = np.zeros((size, size), dtype=float)

        for column in x:
            for row in y:
                index = column * row + row

                z[column, row] = errors[index][0]

        x = np.linspace(0.0, 1.0, size)
        y = np.linspace(0.0, 1.0, size)
        plt.plot()
        plt.contourf(x, y, z)
        plt.show()

    errors = calculate_errors(predictions, targets)

    print("===== ERRORS - Deviation of the predicted target to the actual target =====")
    print("MAX:", errors.max())
    print("MIN:", errors.min())
    print("AVG:", errors.mean())
    print("==========")

    if dimension_count == 2:
        draw_2d_chart(size, errors)
    elif dimension_count == 3:
        draw_3d_chart(size, errors)


# Read data
sensors, targets, distances = test_data_reader.read_test_data("2d_3s", "../")

dimension_count = len(targets[0])
sensor_count = len(distances[0])

# sensors = generator.generate_targets(sensor_count, dimension_count)
# targets, distances = generator.generate_data_matrix(50, dimension_count, sensors)

print(sensors)

print("Dimensions:", dimension_count)
print("Sensors: ", sensor_count)

data_length = len(distances)
data_split = int(data_length * 0.8)
learning_distances, testing_distances = distances[:data_split, :], distances[data_split:, :]
learning_targets, testing_targets = targets[:data_split, :], targets[data_split:, :]

model = build_model(dimension_count, sensor_count)

# Train model
model.fit(learning_distances, learning_targets, epochs=10)

# Test model
test_loss, test_mae, test_mse = model.evaluate(testing_distances, testing_targets)
print("Test MAE:", test_mae, ", Test MSE:", test_mse)

# Plot prediction
predict_shit(model, testing_distances, sensors, testing_targets)

visualize_error(model, sensors, size=10, dimension_count=dimension_count)
