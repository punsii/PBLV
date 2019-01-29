import random


def random_pos(dimension, min_range, max_range):
    pos = []
    for i in range(dimension):
        pos.append(random.randrange(min_range, max_range))
    return pos


def generate(dimension, sensors, output_len, min_range, max_range):
    outputs = []
    for i in range(output_len):
        target = random_pos(dimension, min_range, max_range)
        distances = []
        for j in range(len(sensors)):
            sensor = sensors[j]
            if len(sensor) is not dimension:
                raise Exception('sensors position not matching dimension')
            tmp = 0.0
            for k in range(dimension):
                tmp = tmp + (sensor[k] - target[k])**2
            tmp = tmp**0.5
            distances.append(tmp)
        outputs.append([target, distances])
    return outputs


for data in generate(2, [[0, 0], [0, 5], [5, 0]], 50, -10, 10):
    print("target: " + str(data[0]) + ", distances: " + str(data[1]))
