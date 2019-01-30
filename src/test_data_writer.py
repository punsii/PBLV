from src import generator as gen


# Script will generate random test data and write it to a file.
def writeTestDataToFile(dimensions, sensor_count, count, out_file_name):
    file = open(out_file_name, "w+")  # Create file

    sensors = gen.generate_targets(sensor_count, dimensions, 0.0, 1.0)
    # targets = gen.generate_targets_rectified(count, dimensions, 0.0, 1.0, 0.8)
    targets = gen.generate_targets_normalized(count, dimensions, 0.5, 0.1)

    # Write sensor positions in the first line.
    for sensor in sensors:
        file.write(str(sensor) + ", ")
    file.write("\n")

    for target in targets:
        distances = gen.calculate_distances(target, sensors)
        file.write(str(target) + ", " + str(distances) + "\n")

    file.close()


writeTestDataToFile(
    dimensions=2,
    sensor_count=3,
    count=100000,
    out_file_name="../training_data.txt"
)
writeTestDataToFile(
    dimensions=2,
    sensor_count=3,
    count=100000,
    out_file_name="../test_data.txt"
)
