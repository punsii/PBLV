import numpy as np

# Read test data from the passed file name.
def read_test_data(file_name):
    samples = []
    sensors = []
    with open(file_name) as file:
        index = 0
        for line in file:
            if index == 0:
                sensors = _parse_first_line(line)
            else:
                samples.append(_parse_line(line))
            index = index + 1

    return np.array(samples), np.array(sensors)


def _parse_first_line(line):
    result = []

    line = line.strip()
    parts = line.split("],")

    for part in parts:
        part = part.translate({ord(i): None for i in '[]'})
        if len(part) != 0:
            result.append(_parse_number_array(part))

    return result


def _parse_line(line):
    parts = line.split("], ")

    target_part = parts[0]
    distances_part = parts[1]

    # Remove all brackets from string
    target_part = target_part.translate({ord(i): None for i in '[]'})
    distances_part = distances_part.translate({ord(i): None for i in '[]'})

    target = _parse_number_array(target_part)
    distances = _parse_number_array(distances_part)

    return [target, distances]


def _parse_number_array(string):
    parts = string.split(",")

    result = []
    for part in parts:
        number = float(part.strip())
        result.append(number)

    return result
