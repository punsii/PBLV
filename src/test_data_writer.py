"""
Module for writing generated testdata to file
"""
from src import generator as gen
import numpy as np


def write_test_data_to_file(dimensions, sensor_count, count, file_prefix, directory="./"):
    sensors = gen.generate_targets(sensor_count, dimensions)
    """
    Script will generate random test data and write it to a file.
    """
    targets = gen.generate_targets(count, dimensions)
    distances = gen.apply_sensors_on_targets(targets, sensors)
    with open(f"{directory}/{file_prefix}_sensors.txt", "w+") as file:
        np.savetxt(file, sensors)
    with open(f"{directory}/{file_prefix}_targets.txt", "w+") as file:
        np.savetxt(file, targets)
    with open(f"{directory}/{file_prefix}_distances.txt", "w+") as file:
        np.savetxt(file, distances)


write_test_data_to_file(
    dimensions=2,
    sensor_count=3,
    count=100000,
    file_prefix="2d_3s",
    directory="../"
)
