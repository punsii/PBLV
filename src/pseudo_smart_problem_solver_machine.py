"""
Module for training and evaluating position-estimating neural net
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import test_data_reader
from src import generator



def build_model(dimension_count, sensor_count):
    """
    Configure and compile tensorFlow model.
    """
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


def split_data(data):
    """
    Split data in target and distance array.
    """
    targets = []
    distances = []

    for data_set in data:
        targets.append(data_set[0])
        distances.append(data_set[1])

    return np.array(targets, dtype=float), np.array(distances, dtype=float)

def visualize_shit_interactive(model, sensors):
    """
    Plot an interacitve prediction graph.
    """
    from matplotlib.widgets import Slider

    x_min = 0
    x_max = 1
    x_init = 0.5

    y_min = 0
    y_max = 1
    y_init = 0.5

    test_distances = np.array([generator.calculate_distances(np.array([x_init, y_init]), sensors)])
    predictions = model.predict(test_distances)
    prediction = predictions[0]

    xAxisSensors = []
    yAxisSensors = []
    for sensor_pos in sensors:
        xAxisSensors.append(sensor_pos[0])
        yAxisSensors.append(sensor_pos[1])

    fig = plt.figure()

    # first we create the general layount of the figure
    # with two axes objects: one for the plot of the function
    # and the other for the slider
    main_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

    # in main_ax we plot the function with the initial value of the parameter a
    plt.sca(main_ax)  # select main_ax
    plt.title('prediction (x) vs real position (o)')
    first_plot = plt.plot(
        xAxisSensors, yAxisSensors, "ro",
        x_init, y_init, "bs",
        prediction[0], prediction[1], "g^"
    )
    plt.axis([0.0, 1.0, 0.0, 1.0])

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    x_slider = Slider(slider_ax, 'x', x_min, x_max, valinit=x_init)

    def update(new_x):
        """
        update the canvas when slider was moved
        """
        global prediction

        test_distances = np.array([generator.calculate_distances(np.array([new_x, y_init]), sensors)])
        predictions = model.predict(test_distances)
        prediction = predictions[0]

        fig.clf()
        plt.plot(
            xAxisSensors, yAxisSensors, "ro",
            x_init, y_init, "bs",
            prediction[0], prediction[1], "g^"
        )
        plt.axis([0.0, 1.0, 0.0, 1.0])

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)

        fig.canvas.draw_idle()  # redraw the plot

    # the final step is to specify that the slider needs to
    # execute the above function when its value changes
    x_slider.on_changed(update)

    plt.show()

def visualize_error(model, sensors, size, dimension_count):
    """
    Draw a 2D-heatmap of prediction errors for a (size x size) grid.
    """
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

    def draw_3d_chart(size, errors, plot, z_index=0):
        x_axis = np.arange(size)
        y_axis = np.arange(size)

        matrix = np.zeros((size, size), dtype=float)

        for x in x_axis:
            for y in y_axis:
                matrix[x, y] = errors[x * (size ** 2) + y * size + z_index]

        x_axis = np.linspace(0.0, 1.0, size)
        y_axis = np.linspace(0.0, 1.0, size)

        return plot.contourf(x_axis, y_axis, matrix)

    errors = calculate_errors(predictions, targets)

    print("===== ERRORS - Deviation of the predicted target to the actual target =====")
    print("MAX:", errors.max())
    print("MIN:", errors.min())
    print("AVG:", errors.mean())
    print("==========")

    if dimension_count == 2:
        draw_2d_chart(size, errors)
    elif dimension_count == 3:
        fig = plt.figure(figsize=(1, 1))
        sub_plot = fig.add_subplot(111)

        draw_3d_chart(size, errors, sub_plot)

        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        z_slider = Slider(ax=slider_ax, label="z", valmin=0.0, valmax=1.0, valinit=0.5)

        def update(value):
            draw_3d_chart(size, errors, sub_plot, z_index=int(min(value * size, size)))
            fig.canvas.draw_idle()

        z_slider.on_changed(update)

        plt.show()


# Read data
sensors, targets, distances = test_data_reader.read_test_data("3d_5s", "../")

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
visualize_shit_interactive(model, sensors)

visualize_error(model, sensors, size=10, dimension_count=dimension_count)
