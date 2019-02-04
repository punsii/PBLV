"""
Module for generating random testdata
"""
import numpy as np


def dataset_generator(sensors, dimension_count, batch_size):
    """
    Keras data generator function.
    :param sensors: positions of the sensors
    :param batch_size: size of the batch
    :return:
    """
    while True:
        targets = generate_targets(batch_size, dimension_count)
        distances = apply_sensors_on_targets(targets, sensors)

        yield distances, targets


def generate_data_matrix(size, dimension_count, sensors, range_min=0.0, range_max=1.0):
    """
    Generate a data matrix with the given [sensors] positions.
    Set the resulting data size by setting [size] and [dimension_count].
    The resulting size will be size^dimension_count.

    :param size: of the axis
    :param dimension_count: count of dimensions to generate data for
    :param sensors: positions of the sensors
    :param range_min: the axis ranges minimum value
    :param range_max: the axis ranges maximum value
    :return: n-dimensional matrix filled with data
    """

    # Validate sensor dimensions
    for sensor in sensors:
        if len(sensor) != dimension_count:
            raise Exception(f"Sensor position dimension does not match passed dimension count {dimension_count}")

    axis_values = np.zeros(shape=(dimension_count, size))

    tmp_values = np.linspace(start=range_min, stop=range_max, num=size)
    for i in range(dimension_count):
        axis_values[i] = tmp_values

    targets = []
    distances = []
    _generate_data_matrix_distances(
        axis_values,
        index=0,
        max_index=dimension_count - 1,
        targets=targets,
        distances=distances,
        sensors=sensors,
    )

    targets = np.array(targets)
    distances = np.array(distances)

    return targets, distances


def _generate_data_matrix_distances(axis_values, index, max_index, targets, distances, sensors, *values):
    for value in axis_values[index]:
        if index < max_index:
            _generate_data_matrix_distances(axis_values, index + 1, max_index, targets, distances, sensors, *values,
                                            value)
        else:
            target = np.array([*values, value])
            targets.append(target)

            distances.append(calculate_distances(target, sensors))


def generate_targets(number_of_targets, dimension):
    """
    generates number_of_targets targets.
    """
    return np.random.rand(number_of_targets, dimension)


def shitty_distance(a, b):
    """
    Calculate shitty block distance.
    ( https://de.wikipedia.org/wiki/Manhattan-Metrik )
    """
    return np.sum(np.abs(a - b))


def distance(a, b):
    """
    returns distance between two numpy-arrays.
    """
    return np.linalg.norm(a - b)


def calculate_shitty_distances(target, sensors):
    return np.apply_along_axis(lambda sensor: shitty_distance(target, sensor), 1, sensors)


def calculate_distances(target, sensors):
    """
    calculate distances for a list sensors.
    """
    return np.apply_along_axis(lambda sensor: distance(target, sensor), 1, sensors)


def apply_sensors_on_targets(targets, sensors):
    """
    calculate distances matrix between targets and sensors.
    """
    return np.apply_along_axis(lambda target: calculate_distances(target, sensors), 1, targets)


def shitty_apply_sensors_on_targets(targets, sensors):
    return np.apply_along_axis(lambda target: calculate_shitty_distances(target, sensors), 1, targets)


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
