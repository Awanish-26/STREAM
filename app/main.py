import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64


MODEL_PATH = os.path.join(os.path.dirname(
    __file__), '..', 'models', 'model.keras')


# Define the expected input patch size for the model
PATCH_SIZE = 128

app = Flask(__name__)

# --- Load the Trained Model ---
# Load the custom loss function if your model needs it


def weighted_sce(y_true, y_pred):
    # This is a placeholder. You might need the actual class_weights
    # if you load a model that was compiled with this loss.
    # For simple prediction, it's often not required.
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
                                       'weighted_sce': weighted_sce})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Preprocessing and Prediction Functions ---


def preprocess_input(elevation_data):
    """Normalizes a single elevation channel to 0-1."""
    norm_elevation = (elevation_data - np.min(elevation_data)) / \
        (np.max(elevation_data) - np.min(elevation_data) + 1e-8)

    # The model expects 3 channels (elevation, slope, roughness).
    # For this demo, we will stack the elevation channel 3 times.
    input_stack = np.stack(
        [norm_elevation, norm_elevation, norm_elevation], axis=-1)

    # Add a batch dimension
    return np.expand_dims(input_stack, axis=0)


def postprocess_output(prediction):
    """Converts model output to an image."""
    # Get the class with the highest probability for each pixel
    pred_classes = np.argmax(prediction[0], axis=-1).astype(np.uint8)

    # Define the color map: 0=Red (Unsafe), 1=Orange (Moderate), 2=Green (Safe)
    color_map = {
        0: [255, 0, 0],    # Red
        1: [255, 165, 0],  # Orange
        2: [0, 255, 0]     # Green
    }

    # Create an RGB image from the prediction classes
    h, w = pred_classes.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in color_map.items():
        rgb_image[pred_classes == c] = color

    # Convert to a format that can be sent to the frontend
    img = Image.fromarray(rgb_image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# --- Index Route ---
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives an elevation patch, runs prediction, and returns a safety map."""
    if not model:
        return jsonify({'error': 'Model is not loaded!'}), 500

    # For this demo, we'll generate a random patch of data
    # In a real app, you would get this from a user upload or a map click
    random_elevation_patch = np.random.rand(
        PATCH_SIZE, PATCH_SIZE) * 1000  # Random elevation data

    # Preprocess, predict, and postprocess
    processed_input = preprocess_input(random_elevation_patch)
    prediction = model.predict(processed_input)
    output_image_str = postprocess_output(prediction)

    return jsonify({'prediction_map': output_image_str})


if __name__ == '__main__':
    app.run(debug=True)
