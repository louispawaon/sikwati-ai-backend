from flask import Flask, render_template, request, jsonify
from app import app

import os
import joblib
import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

model = keras.models.load_model("app/models/efficientnet.h5")
scaler = joblib.load("app/scaler/scaler.pkl")

class_names = ["black_pod_rot", "frosty_pod_rot", "healthy", "pod_borer"]


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/classify_image", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"})

    uploaded_image = request.files["image"]
    if uploaded_image.filename == "":
        return jsonify({"error": "No selected file"})

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
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction).item()

        return jsonify({"class": predicted_class, "confidence": confidence})
