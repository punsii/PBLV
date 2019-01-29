import random


def generate_target(dimension, min_range, max_range):
    pos = []
    for i in range(dimension):
        pos.append(random.randrange(min_range, max_range))
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
        tmp = tmp + (a[i] - b[i])**2
    tmp = tmp**0.5
    return tmp


def calculate_distances(target, sensors):
    distances = []
    for sensor in sensors:
        distances.append(distance(target, sensor))
    return distances


number_of_targets = 50000
dimension = 3
min_range = -5
max_range = 5
sensors = [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]
targets = generate_targets(number_of_targets, dimension, min_range, max_range)


for target in targets:
    distances = calculate_distances(target, sensors)
    print(f"target: {target}, distances: {distances}")
