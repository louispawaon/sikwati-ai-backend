from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app import app

import os
import joblib
import numpy as np
from PIL import Image
from tensorflow import keras
import tflite_runtime.interpreter as tflite
from sklearn.preprocessing import StandardScaler

# Load the H5 Model
h5_model = keras.models.load_model("app/models/efficientnet.h5")

scaler = joblib.load("app/scaler/scaler.pkl")

# # Load the TFLite model
# interpreter = tflite.Interpreter(model_path="app/models/efficientnet_mobile.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

class_names = ["black_pod_rot", "frosty_pod_rot", "healthy", "pod_borer"]


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/api/classify_image", methods=["POST"])
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
        prediction = h5_model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction).item()

        return jsonify({"class": predicted_class, "confidence": confidence})


# @app.route("/api/mobile_classify", methods=["POST"])
# def mobile_classify():
#     if "image" not in request.files:
#         return jsonify({"error": "No file part"})

#     uploaded_image = request.files["image"]
#     if uploaded_image.filename == "":
#         return jsonify({"error": "No selected file"})

#     if uploaded_image:
#         # Read and preprocess the image
#         image = Image.open(uploaded_image)
#         image = image.resize((224, 224))  # Resize the image
#         image = np.array(image)
#         image = scaler.transform(image.reshape(1, -1)).reshape(
#             image.shape
#         )  # Standardization

#         # Make sure the input shape matches the TFLite model's input shape
#         input_shape = input_details[0]["shape"]
#         image = image.astype(np.float32)
#         image = np.expand_dims(image, axis=0)
#         if not np.all(image.shape == input_shape):
#             return jsonify(
#                 {"error": "Input image shape does not match model input shape"}
#             )

#         # Perform inference
#         interpreter.set_tensor(input_details[0]["index"], image)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]["index"])

#         predicted_class = class_names[np.argmax(output_data)]
#         confidence = np.max(output_data).item()

#         return jsonify({"class": predicted_class, "confidence": confidence})
