"""
Module for generating random testdata
"""
import numpy as np


def dataset_generator(sensors, dimension_count, batch_size, cityblock=False):
    """
    Keras data generator function.
    :param sensors: positions of the sensors
    :param batch_size: size of the batch
    :return:
    """
    while True:
        targets = generate_targets(batch_size, dimension_count)
        distances = apply_sensors_on_targets(targets, sensors, cityblock)

        yield distances, targets


def generate_data_matrix(size, dimension_count, sensors, range_min=0.0, range_max=1.0, cityblock=False):
    """
    Generate a data matrix with the given [sensors] positions.
    Set the resulting data size by setting [size] and [dimension_count].
    The resulting size will be size^dimension_count.

    :param size: of the axis
    :param dimension_count: count of dimensions to generate data for
    :param sensors: positions of the sensors
    :param range_min: the axis ranges minimum value
    :param range_max: the axis ranges maximum value
    :param cityblock: whether to use cityblock metrics
    :return: n-dimensional matrix filled with data
    """
    # Validate sensor dimensions
    for sensor in sensors:
        if len(sensor) != dimension_count:
            raise Exception(f"Sensor position dimension does not match passed dimension count {dimension_count}")

    sensor_count = len(sensors)

    axis_values = np.linspace(start=range_min, stop=range_max, num=size)

    count = size ** dimension_count
    counter = np.zeros(shape=dimension_count, dtype=int)
    max_counter_index = size - 1

    targets = np.zeros(shape=(count, dimension_count))
    distances = np.zeros(shape=(count, len(sensors)))

    for i in range(count):
        for dimension_index in range(dimension_count):
            targets[i, dimension_index] = axis_values[counter[dimension_index]]

        for sensor_index in range(sensor_count):
            distances[i, sensor_index] = distance(targets[i], sensors[sensor_index], cityblock)

        counter_index = dimension_count - 1
        while True:
            if counter[counter_index] < max_counter_index:
                counter[counter_index] += 1
                break
            else:
                counter[counter_index] = 0
                counter_index -= 1

    return targets, distances


def generate_targets(number_of_targets, dimension):
    """
    generates number_of_targets targets.
    """
    return np.random.rand(number_of_targets, dimension)


def distance(a, b, cityblock=False):
    """
    returns distance between two numpy-arrays.
    """
    return np.linalg.norm(a - b) if not cityblock else np.sum(np.abs(a - b))


def calculate_distances(target, sensors, cityblock=False):
    """
    calculate distances for a list sensors.
    """
    return np.apply_along_axis(lambda sensor: distance(target, sensor, cityblock), 1, sensors)


def apply_sensors_on_targets(targets, sensors, cityblock=False):
    """
    calculate distances matrix between targets and sensors.
    """
    return np.apply_along_axis(lambda target: calculate_distances(target, sensors, cityblock), 1, targets)


def test():
    """
    tesfunction
    """
    number_of_targets = 20
    dimension = 2
    sensors = generate_targets(3, 2)
    targets = generate_targets(number_of_targets, dimension)
    distances = apply_sensors_on_targets(targets, sensors)

    for i in range(len(targets)):
        print(f"target: {targets[i]}, distances: {distances[i]}")
