# Read test data from the passed file name.
def read_test_data(file_name):
    samples = []
    with open(file_name) as file:
        for line in file:
            samples.append(_parse_line(line))

    return samples


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
        number = float(part)
        result.append(number)

    return result
