"""
Module for training and evaluating position-estimating neural net
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
import plotly.plotly as py
import plotly.tools as tls
import numpy as np
import tensorflow as tf
from tensorflow import keras

import generator


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


def build_model(dimension_count, sensor_count):
    """
    Configure and compile tensorFlow model.
    """
    model = keras.Sequential([
        keras.layers.Dense(16 * dimension_count, activation=tf.nn.elu,
                           input_shape=(sensor_count,)),
        keras.layers.Dense(8 * dimension_count, activation=tf.nn.elu),
        keras.layers.Dense(4 * dimension_count, activation=tf.nn.elu),
        keras.layers.Dense(2 * dimension_count, activation=tf.nn.elu),
        keras.layers.Dense(dimension_count, activation=tf.nn.relu)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", "mse"]
    )

    return model


def train_model(dimension_count, sensor_count, batch_size, steps, validation_steps, epochs, city_block=False):
    model = build_model(dimension_count, sensor_count)

    sensors = generator.generate_targets(sensor_count, dimension_count)

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='../log',
                                                histogram_freq=1,
                                                write_graph=True,
                                                write_grads=True,
                                                write_images=True,
                                                batch_size=32)

    model.fit_generator(
        generator=generator.dataset_generator(sensors=sensors,
                                              dimension_count=dimension_count,
                                              batch_size=batch_size,
                                              cityblock=city_block),
        steps_per_epoch=steps,
        validation_data=generator.dataset_generator(sensors=sensors,
                                                    dimension_count=dimension_count,
                                                    batch_size=batch_size,
                                                    cityblock=city_block),
        validation_steps=validation_steps,
        callbacks=[tbCallBack],
        epochs=epochs
    )

    return model, sensors


def visualize_error_per_dimension(model, sensors, dimension_count):
    """
    Plot average error for each dimension
    """
    def draw(example_generator):
        plt.clf()
        distances, targets = next(example_generator)
        predictions = model.predict(distances)
        error = np.absolute(targets[0] - predictions[0])
        plt.bar(np.arange(dimension_count)-0.2, targets[0],
                width=0.2, color='green', label='target')
        plt.bar(np.arange(dimension_count), predictions[0],
                width=0.2, color='blue', label='prediction')
        plt.bar(np.arange(dimension_count)+0.2, error,
                width=0.2, color='red', label='error')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=3, fancybox=True, shadow=True)
        plt.draw()

    example_generator = generator.dataset_generator(sensors, dimension_count, 1)

    fig, _ = plt.subplots()
    plt.xlabel("dimension")
    plt.ylabel("absolute error")
    plt.title("error per dimension")
    fig.canvas.mpl_connect('button_press_event',
                           lambda event: draw(example_generator))

    draw(example_generator)
    plt.show()

def visualize_2D_3D_interactive(model, sensors, size, dimension_count):
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
                                     errors.transpose(), cmap=cm.rainbow)

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
        errors[i] = generator.distance(predictions[i], targets[i])

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

        def slider_update(subplot, cur_pos, value):
            cur_pos[2] = value

            draw_plot(subplot, size, errors, cur_pos)

        z_slider.on_changed(lambda z_val: slider_update(subplot, cur_pos, z_val))

    plt.show()
