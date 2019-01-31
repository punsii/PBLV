"""
Module for reading sensor, target and distance data from file
"""
import numpy as np


def read_test_data(file_prefix, directory="./"):
    """
    Read test data from the passed file name.
    """
    sensors = np.loadtxt(f"{directory}{file_prefix}_sensors.txt")
    targets = np.loadtxt(f"{directory}{file_prefix}_targets.txt")
    distances = np.loadtxt(f"{directory}{file_prefix}_distances.txt")
    return sensors, targets, distances
