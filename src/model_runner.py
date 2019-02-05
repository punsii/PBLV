import os
from argparse import ArgumentParser
from pseudo_smart_problem_solver_machine import train_model, visualize_2D_3D_interactive, visualize_error_per_dimension
import tensorflow as tf
import numpy as np
import json


def load_configuration(configuration_file_path, args):
    with open(configuration_file_path, "r") as file:
        content = file.read()

        configuration_dictionary = json.loads(content)

        if "visualize_errors" in configuration_dictionary:
            args.visualize_errors = configuration_dictionary["visualize_errors"]
        if "error_visualization_accuracy" in configuration_dictionary:
            args.error_visualization_accuracy = configuration_dictionary["error_visualization_accuracy"]
        if "dimension_count" in configuration_dictionary:
            args.dimension_count = configuration_dictionary["dimension_count"]
        if "sensor_count" in configuration_dictionary:
            args.sensor_count = configuration_dictionary["sensor_count"]
        if "epochs" in configuration_dictionary:
            args.epochs = configuration_dictionary["epochs"]
        if "steps" in configuration_dictionary:
            args.steps = configuration_dictionary["steps"]
        if "validation_steps" in configuration_dictionary:
            args.validation_steps = configuration_dictionary["validation_steps"]
        if "batch_size" in configuration_dictionary:
            args.batch_size = configuration_dictionary["batch_size"]


def store_configuration(configuration_file_path, args, omit_visualize_errors=False):
    with open(configuration_file_path, "w+") as file:
        configuration_dictionary = {
            "error_visualization_accuracy": args.error_visualization_accuracy,
            "dimension_count": args.dimension_count,
            "sensor_count": args.sensor_count,
            "epochs": args.epochs,
            "steps": args.steps,
            "validation_steps": args.validation_steps,
            "batch_size": args.batch_size,
        }

        if not omit_visualize_errors:
            configuration_dictionary["visualize_errors"] = args.visualize_errors

        file.write(json.dumps(configuration_dictionary, indent=4))


parser = ArgumentParser()
parser.add_argument("-c", "--config", dest="configuration",
                    help="Path to a configuration file to use")
parser.add_argument("-s", "--save", dest="save_model", action="store_true",
                    help="Whether the model should be saved to folder defined in --model_path")
parser.add_argument("-l", "--load", dest="load_model", action="store_true",
                    help="Whether the model defined in --model_path should be loaded")
parser.add_argument("-f", "--model_path", dest="model_path", default="model_output/default",
                    help="A model file on disk to load from or save to")
parser.add_argument("-v", "--visualize_errors", dest="visualize_errors", action="store_true",
                    help="Visualize the resulting models errors")
parser.add_argument("--error_visualization_accuracy", dest="error_visualization_accuracy", type=int, default=50,
                    help="The resolution of the error visualization")
parser.add_argument("--dimension_count", dest="dimension_count", type=int, default=3,
                    help="How many dimensions the model should be configured and trained with")
parser.add_argument("--sensor_count", dest="sensor_count", type=int, default=4,
                    help="How many sensors should be placed randomly in the \"room\"")
parser.add_argument("--epochs", dest="epochs", type=int,
                    default=10, help="How many epochs to train the model")
parser.add_argument("--steps", dest="steps", type=int,
                    default=1000, help="Steps to train the model")
parser.add_argument("--validation_steps", dest="validation_steps", default=200, type=int,
                    help="Steps to validate the model after training with")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=100,
                    help="Batch size of single steps to train the model with")

args = parser.parse_args()

if args.configuration:
    load_configuration(args.configuration, args)

MODEL = None
SENSORS = None

print("===== Running model runner with arguments: =====")
if args.save_model or args.load_model:
    saved_loaded_expr = ""
    if args.save_model:
        saved_loaded_expr = "saved to"
    else:
        saved_loaded_expr = "loaded from"
    print("Model will be", saved_loaded_expr, f"\"{args.model_path}\"")
print("================================================\n")

if args.load_model:
    print("========= Loading the model from file ==========")
    MODEL = tf.keras.models.load_model(
        f"{args.model_path}/model",
        custom_objects=None,
        compile=True
    )

    # Load sensors
    SENSORS = np.loadtxt(f"{args.model_path}/sensors")

    # Load configuration
    if not args.configuration:
        # Only load configuration if --config is not specified
        load_configuration(f"{args.model_path}/configuration", args)
    print("===================== DONE =====================\n")

# Print final configuration
print("================ Configuration =================")
print("Visualize errors:", args.visualize_errors)
print("Error visualization accuracy:", args.error_visualization_accuracy)
print("Dimension count:", args.dimension_count)
print("Sensor count:", args.sensor_count)
print("Epochs:", args.epochs)
print("Steps:", args.steps)
print("Validation Steps:", args.validation_steps)
print("Batch size:", args.batch_size)
print("================================================\n")

if not args.load_model:
    print("======== Training the model for you <3 =========")
    MODEL, SENSORS = train_model(
        dimension_count=args.dimension_count,
        sensor_count=args.sensor_count,
        batch_size=args.batch_size,
        steps=args.steps,
        validation_steps=args.validation_steps,
        epochs=args.epochs
    )
    print("===================== DONE =====================\n")

if args.save_model:
    print("=========== Storing the trained model ==========")
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    tf.keras.models.save_model(
        MODEL,
        f"{args.model_path}/model",
        overwrite=True,
        include_optimizer=True
    )

    # Store sensors
    with open(f"{args.model_path}/sensors", "w+") as file:
        np.savetxt(file, SENSORS)

    # Store configuration
    store_configuration(f"{args.model_path}/configuration", args, omit_visualize_errors=True)

    print("===================== DONE =====================\n")

if args.visualize_errors:
    print("========= Visualizing the model errors =========")
    if 2 <= args.dimension_count <= 3:
        visualize_2D_3D_interactive(
            MODEL, SENSORS, args.error_visualization_accuracy, args.dimension_count)
    else:
        visualize_error_per_dimension(
            MODEL, SENSORS, args.dimension_count)
    print("===================== DONE =====================\n")

print("=========== Model runner terminated ============")
