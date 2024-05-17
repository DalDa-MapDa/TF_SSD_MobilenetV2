import tensorflow as tf
import os

def generate_labels_from_tflite(model_path, output_labels_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get details of the output tensors
    output_details = interpreter.get_output_details()

    # Extract number of output classes
    output_tensor = interpreter.tensor(output_details[0]['index'])()
    num_classes = output_tensor.shape[-1]

    # Generate basic labels
    labels = [f"Label_{i}" for i in range(num_classes)]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

    # Save labels to a file
    with open(output_labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Labels saved to {output_labels_path}")

# Example usage with os.path for proper path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19', 'model.tflite')
output_labels_path = os.path.join(current_dir, 'txt_output', 'labels.txt')

generate_labels_from_tflite(model_path, output_labels_path)
