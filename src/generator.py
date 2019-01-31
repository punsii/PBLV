import numpy as np


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
    return np.random.rand(number_of_targets, dimension)


def distance(a, b):
    return np.linalg.norm(a - b)


def calculate_distances(target, sensors):
    return np.apply_along_axis(lambda sensor: distance(target, sensor), 1, sensors)


def apply_sensors_on_targets(targets, sensors):
    return np.apply_along_axis(lambda target: calculate_distances(target, sensors), 1, targets)


def test():
    number_of_targets = 20
    dimension = 2
    sensors = generate_targets(3, 2)
    targets = generate_targets(number_of_targets, dimension)
    distances = apply_sensors_on_targets(targets, sensors)

    for i in range(len(targets)):
        print(f"target: {targets[i]}, distances: {distances[i]}")
