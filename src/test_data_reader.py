import numpy as np


# Read test data from the passed file name.
def read_test_data(file_prefix, directory="./"):
    sensors = np.loadtxt(f"{directory}{file_prefix}_sensors.txt")
    targets = np.loadtxt(f"{directory}{file_prefix}_targets.txt")
    distances = np.loadtxt(f"{directory}{file_prefix}_distances.txt")
    return sensors, targets, distances
