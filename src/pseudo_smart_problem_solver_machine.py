"""
Module for training and evaluating position-estimating neural net
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
import numpy as np
import tensorflow as tf
from tensorflow import keras

import test_data_reader
import generator


def build_model(dimension_count, sensor_count):
    """
    Configure and compile tensorFlow model.
    """
    model = keras.Sequential([
        keras.layers.Dense(20 * sensor_count, activation=tf.nn.relu,
                           input_shape=(sensor_count,)),
        keras.layers.Dropout(0.75),
        keras.layers.Dense(20 * sensor_count, activation=tf.nn.softmax),
        keras.layers.Dropout(0.75),
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


# def visualize_shit_interactive(model, sensors):
#    """
#    Plot an interacitve prediction graph.
#    """
#
#    def update(subplot, new_x, new_y):
#        """
#        update the canvas when slider was moved
#        """
#        test_distances = np.array([
#            generator.calculate_distances(np.array([new_x, new_y]), sensors)
#        ])
#        predictions = model.predict(test_distances)
#        prediction = predictions[0]
#
#        subplot.clear()
#        subplot.plot(
#            x_axis_sensors, y_axis_sensors, "g^",
#            new_x, new_y, "bo",
#            prediction[0], prediction[1], "rx"
#        )
#
#        subplot.axis([0.0, 1.0, 0.0, 1.0])
#        plt.xlim(0, 1)
#        plt.ylim(0, 1)
#
#        fig.canvas.draw_idle()  # redraw the plot
#
#    x_axis_sensors = []
#    y_axis_sensors = []
#    for sensor_pos in sensors:
#        x_axis_sensors.append(sensor_pos[0])
#        y_axis_sensors.append(sensor_pos[1])
#
#   # Plot
#   fig = plt.figure()
#   plt.title('prediction (x) vs real position (o)')
#   subplot = fig.add_subplot(111)
#   update(subplot, 0.5, 0.5)
#
#    # Define x-slider
#    x_axis = plt.axes([0.125, 0.01, 0.78, 0.04])
#    y_axis = plt.axes([0, 0.3, 0.04, 0.78])
#    x_slider = Slider(x_axis, 'x', 0, 1, valinit=0.5)
#    y_slider = Slider(y_axis, 'x', 0, 1, valinit=0.5)
#
#   def onklick(event):
#       update(subplot, event.xdata, event.ydata)
#
#   fig.canvas.mpl_connect('button_press_event', onklick)
#
#   # Let's rock
#    y_slider.on_changed(lambda y_val: update(subplot, x_slider.val, y_val))
#    x_slider.on_changed(lambda x_val: update(subplot, x_val, y_slider.val))
#    plt.show()


def visualize_error_interactive(model, sensors, size, dimension_count):
    """
    Draw a 2D-heatmap of prediction errors for a (size x size) grid.
    """

    def draw_plot(subplot, size, errors, cur_pos):
        if len(errors.shape) == 3:
            # Take 2D slice of 3D data
            errors = errors[:, :, int(cur_pos[2] * (size - 1))]

        subplot.clear()
        cont_plot = subplot.contourf(np.linspace(0.0, 1.0, size),
                                     np.linspace(0.0, 1.0, size),
                                     errors.transpose(), cmap=cm.PuBu_r)

        example_distances = np.array([
            generator.calculate_distances(np.array(cur_pos), sensors)
        ])
        prediction = model.predict(example_distances)[0]

        subplot.plot(
            cur_pos[0], cur_pos[1], "ko",
            prediction[0], prediction[1], "rx"
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        subplot.axis([0.0, 1.0, 0.0, 1.0])

        fig.canvas.draw_idle()  # redraw the plot
        return cont_plot

    # Prepare data first
    targets, distances = \
        generator.generate_data_matrix(size, dimension_count, sensors)

    predictions = model.predict(distances)
    errors = np.zeros(predictions.shape[0])
    for i, _ in enumerate(predictions):
        errors[i] = np.linalg.norm(
            predictions[i] - targets[i])

    if dimension_count == 2:
        errors = np.reshape(errors, (size, size))
        cur_pos = [0.5, 0.5]
    elif dimension_count == 3:
        cur_pos = [0.5, 0.5, 0]
        errors = np.reshape(errors, (size, size, size))

    print(cur_pos)

    print("===== ERRORS - Deviation of the predicted target to the actual target =====")
    print("MAX:", errors.max())
    print("MIN:", errors.min())
    print("AVG:", errors.mean())
    print("==========")

    # Prepare Plot
    fig = plt.figure()
    plt.title('prediction (x) vs real position (o) + heatmap')
    subplot = fig.add_subplot(111)
    cont_plot = draw_plot(subplot, size, errors, cur_pos)
    fig.colorbar(cont_plot)

    # Define mouse interaction
    def onklick(event):
        cur_pos[0] = event.xdata
        cur_pos[1] = event.ydata

        draw_plot(subplot, size, errors, cur_pos)

    fig.canvas.mpl_connect('button_press_event', onklick)

    # Slider for z-axis if present
    if dimension_count == 3:
        slider_ax = plt.axes([0.125, 0.01, 0.78, 0.04])
        z_slider = Slider(ax=slider_ax, label="z",
                          valmin=0.0, valmax=1.0, valinit=0)

        def update(subplot, cur_pos, value):
            cur_pos[2] = value

            draw_plot(subplot, size, errors, cur_pos)

        z_slider.on_changed(lambda z_val: update(subplot, cur_pos, z_val))

    plt.show()


def dataset_generator(sensors, dimension_count, batch_size):
    """
    Keras data generator function.
    :param sensors: positions of the sensors
    :param batch_size: size of the batch
    :return:
    """
    while True:
        targets = generator.generate_targets(batch_size, dimension_count)
        distances = generator.apply_sensors_on_targets(targets, sensors)

        yield distances, targets


def train_model(dimension_count, sensor_count, batch_size, steps, validation_steps, epochs):
    model = build_model(dimension_count, sensor_count)

    sensors = generator.generate_targets(sensor_count, dimension_count)

    model.fit_generator(
        generator=dataset_generator(sensors=sensors, dimension_count=dimension_count, batch_size=batch_size),
        steps_per_epoch=steps,
        validation_data=dataset_generator(sensors=sensors, dimension_count=dimension_count, batch_size=batch_size),
        validation_steps=validation_steps,
        epochs=epochs
    )

    return model, sensors


dimension_count = 10
model, sensors = train_model(
    dimension_count=dimension_count,
    sensor_count=11,
    batch_size=100,
    steps=100000,
    validation_steps=200,
    epochs=10
)

# Plot prediction for 2D or 3D data
if 2 <= dimension_count <= 3:
    visualize_error_interactive(model, sensors, 60, dimension_count)
