from argparse import ArgumentParser
from pseudo_smart_problem_solver_machine import train_model, visualize_2D_3D_interactive, visualize_error_per_dimension
import tensorflow as tf
import numpy as np

parser = ArgumentParser()
parser.add_argument("-s", "--save", dest="save_model", action="store_true",
                    help="Whether the model should be saved to file defined in --model_path")
parser.add_argument("-l", "--load", dest="load_model", action="store_true",
                    help="Whether the model defined in --model_path should be loaded")
parser.add_argument("-f", "--model_path", dest="model_path", default="last.model",
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

print("===== Running model runner with arguments: =====")
print("Save model?", args.save_model)
print("Load model?", args.load_model)
print("Save/Load model from/to", f"\"{args.model_path}\"")
print("Visualize errors:", args.visualize_errors)
print("Error visualization accuracy:", args.error_visualization_accuracy)
print("Dimension count:", args.dimension_count)
print("Sensor count:", args.sensor_count)
print("Epochs:", args.epochs)
print("Steps:", args.steps)
print("Validation Steps:", args.validation_steps)
print("Batch size:", args.batch_size)
print("================================================\n\n")

MODEL = None
SENSORS = None

if args.load_model:
    print("========= Loading the model from file ==========")
    MODEL = tf.keras.models.load_model(
        args.model_path,
        custom_objects=None,
        compile=True
    )

    # Load sensors
    SENSORS = np.loadtxt(f"{args.model_path}.sensors")
    print("===================== DONE =====================\n\n")
else:
    print("==== Configuring/Training the model for you ====")
    MODEL, SENSORS = train_model(
        dimension_count=args.dimension_count,
        sensor_count=args.sensor_count,
        batch_size=args.batch_size,
        steps=args.steps,
        validation_steps=args.validation_steps,
        epochs=args.epochs
    )
    print("===================== DONE =====================\n\n")

if args.save_model:
    print("=========== Storing the trained model ==========")
    tf.keras.models.save_model(
        MODEL,
        args.model_path,
        overwrite=True,
        include_optimizer=True
    )

    # Store sensors
    with open(f"{args.model_path}.sensors", "w+") as file:
        np.savetxt(file, SENSORS)
    print("===================== DONE =====================\n\n")

if args.visualize_errors:
    print("========= Visualizing the model errors =========")
    if 2 <= args.dimension_count <= 3:
        visualize_2D_3D_interactive(
            MODEL, SENSORS, args.error_visualization_accuracy, args.dimension_count)
    else:
        visualize_error_per_dimension(
            MODEL, SENSORS, args.error_visualization_accuracy, args.dimension_count)
    print("===================== DONE =====================\n\n")

print("=========== Model runner terminated ============")
