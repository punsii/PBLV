import random


def generate_target_rectified(dimension, min_range, max_range, percentage_of_border_targets):
    if random.uniform(0.0, 1.0) > percentage_of_border_targets:
        return generate_target(dimension, min_range, max_range)
    n = random.randrange(0, dimension)
    pos = []
    for i in range(dimension):
        if i is n:
            pos.append(random.uniform(min_range, max_range))
        else:
            if random.uniform(0.0, 1.0) >= 0.5:
                pos.append(max_range)
            else:
                pos.append(min_range)
    return pos


def generate_targets_rectified(number_of_targets, dimension, min_range, max_range, percentage_of_border_targets):
    targets = []
    for i in range(number_of_targets):
        targets.append(generate_target_rectified(dimension, min_range, max_range, percentage_of_border_targets))
    return targets


def generate_target_normalized(dimension, mu, sigma):
    pos = []
    for i in range(dimension):
        pos.append(random.normalvariate(mu, sigma))
    return pos


def generate_targets_normalized(number_of_targets, dimension, mu, sigma):
    targets = []
    for i in range(number_of_targets):
        targets.append(generate_target_normalized(dimension, mu, sigma))
    return targets


def generate_target(dimension, min_range, max_range):
    pos = []
    for i in range(dimension):
        pos.append(random.uniform(min_range, max_range))
    return pos


def generate_targets(number_of_targets, dimension, min_range, max_range):
    targets = []
    for i in range(number_of_targets):
        targets.append(generate_target(dimension, min_range, max_range))
    return targets


def distance(a, b):
    if len(a) is not len(b):
        raise Exception('Dimensions not matching')
    tmp = 0.0
    for i in range(len(a)):
        tmp = tmp + (a[i] - b[i]) ** 2
    tmp = tmp ** 0.5
    return tmp


def calculate_distances(target, sensors):
    distances = []
    for sensor in sensors:
        distances.append(distance(target, sensor))
    return distances


def test():
    number_of_targets = 50000
    dimension = 3
    min_range = -5
    max_range = 5
    sensors = [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]
    targets = generate_targets(number_of_targets, dimension, min_range, max_range)

    for target in targets:
        distances = calculate_distances(target, sensors)
        print(f"target: {target}, distances: {distances}")
