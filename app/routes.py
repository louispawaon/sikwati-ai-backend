from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app import app

import os
import joblib
import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from skimage import exposure


# Load the H5 Model, PKL File
h5_model = keras.models.load_model("app/models/efficientnet.h5")
scaler = joblib.load("app/scaler/scaler_new.pkl")

# Initialize classification names
class_names = ["black_pod_rot", "frosty_pod_rot", "healthy", "pod_borer"]


@app.route("/")
def index():
    """
    Home endpoint.

    ---
    responses:
      200:
        description: Welcome to the Sikwati AI API
    """
    return jsonify("Welcome to the Sikwati AI API"), 200


@app.route("/api/classify_image", methods=["POST"])
def classify_image():
    """
    Endpoint to classify an uploaded image.

    ---
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The image file to classify.

    responses:
      200:
        description: Classification result.
        schema:
          properties:
            class:
              type: string
              description: Predicted class.
            confidence:
              type: number
              format: float
              description: Confidence score.
      400:
        description: Invalid request or image cannot be classified.
    """
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    uploaded_image = request.files["image"]
    if uploaded_image.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if uploaded_image:
        # Read and preprocess the image
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))  # Resize the image
        image = np.array(image)
        image = scaler.transform(image.reshape(1, -1)).reshape(
            image.shape
        )  # Standardize the image
        image = np.expand_dims(image, axis=0)

        # Perform inference
        prediction = h5_model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction).item()

        if predicted_class not in class_names or confidence < 0.50:
            return jsonify({"error": "The image may not contain a cacao pod"}), 400

        return jsonify({"class": predicted_class, "confidence": confidence})
