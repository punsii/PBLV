from src import test_data_reader

import matplotlib.pyplot as plt
import numpy as np
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

samples = data[0]
sensors = data[1]

print(sensors)

dimension_count = len(samples[0][0])
sensor_count = len(samples[0][1])

splitted_data = split_data(samples)
targets = np.array(splitted_data[0], dtype=float)
distances = np.array(splitted_data[1], dtype=float)

print("Dimensions:", dimension_count)
print("Sensors: ", sensor_count)

model = build_model(dimension_count, sensor_count)

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='../log',
                                            histogram_freq=1,
                                            write_graph=True,
                                            write_grads=True,
                                            write_images=True,
                                            batch_size=32)
# Train model
model.fit(distances, targets, epochs=10, callbacks=[tbCallBack])

# Test model
test_data = test_data_reader.read_test_data(file_name="../test_data.txt")

test_samples = test_data[0]
test_sensors = test_data[1]

splitted_data = split_data(test_samples)
test_targets = np.array(splitted_data[0], dtype=float)
test_distances = np.array(splitted_data[1], dtype=float)

test_loss, test_mae, test_mse = model.evaluate(test_distances, test_targets)
print("Test MAE:", test_mae, ", Test MSE:", test_mse)

# Plot prediction
predictions = model.predict(test_distances)
print(predictions[0])

xAxis = []
yAxis = []
for sensor_pos in test_sensors:
    xAxis.append(sensor_pos[0])
    yAxis.append(sensor_pos[1])

plt.plot(xAxis, yAxis, "ro", [test_targets[0][0]], [test_targets[0][1]], "bs", [predictions[0][0]], [predictions[0][1]], "g^")
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()
