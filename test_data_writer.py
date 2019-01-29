import generator as gen
import random


# Script will generate random test data and write it to a file.
def writeTestDataToFile(dimensions, sensor_count, count, out_file_name):
    file = open(out_file_name, "w+")  # Create file

    sensorPositions = generateSensorPositions(sensor_count)

    for data in gen.generate(dimensions, sensorPositions, count, 0.0, 1.0):
        file.write(str(data[0]) + "," + str(data[1]) + "\n")

    file.close()


def generateSensorPositions(count):
    result = []

    for i in range(count):
        result.append([
            random.uniform(0.0, 1.0),
            random.uniform(0.0, 1.0)
        ])

    return result


writeTestDataToFile(
    dimensions=2,
    sensor_count=3,
    count=10000,
    out_file_name="test_data.txt"
)
