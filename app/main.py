from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import rasterio
from scipy.ndimage import label

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model("../models/model.keras")

# Load DEM
with rasterio.open("mars_dem.tif") as src:
    dem = src.read(1)

PATCH_SIZE = 128
STRIDE = 32


def generate_landing_map(region):

    height, width = region.shape

    prediction_map = np.zeros((height, width))
    count_map = np.zeros((height, width))

    for i in range(0, height-PATCH_SIZE, STRIDE):
        for j in range(0, width-PATCH_SIZE, STRIDE):

            patch = region[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patch = np.expand_dims(patch, axis=0)
            patch = np.expand_dims(patch, axis=-1)

            pred = model.predict(patch, verbose=0)[0, :, :, 0]

            prediction_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += pred
            count_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += 1

    prediction_map /= count_map

    return prediction_map


@app.route("/landing", methods=["POST"])
def landing():

    data = request.json

    x1 = data["x1"]
    y1 = data["y1"]
    x2 = data["x2"]
    y2 = data["y2"]

    region = dem[y1:y2, x1:x2]

    safety_map = generate_landing_map(region)

    safe_map = safety_map > 0.6

    labeled, num = label(safe_map)

    sizes = [(i, np.sum(labeled == i)) for i in range(1, num+1)]
    sizes.sort(key=lambda x: x[1], reverse=True)

    best_region = sizes[0][0]

    coords = np.column_stack(np.where(labeled == best_region))

    cy = int(np.mean(coords[:, 0]))
    cx = int(np.mean(coords[:, 1]))

    return jsonify({
        "landing_x": int(cx),
        "landing_y": int(cy)
    })


app.run()
