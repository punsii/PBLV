import numpy as np


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
