import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf

app = Flask(__name__)

# Define paths
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model and print its input shape
model_path = os.path.join(STATIC_FOLDER, "models", "model2_catsVSdogs_10epoch_weights.h5")
cnn_model = tf.keras.models.load_model(model_path)
cnn_model.summary()

# Assuming the model expects 128x128 images
IMAGE_SIZE = 128

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    prob = model.predict(preprocessed_image)
    label = "Cat" if prob[0][0] >= 0.5 else "Dog"
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    return label, classified_prob

# Home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

# Handle image upload and classification
@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        # Check if the file part exists in the request
        if "image" not in request.files:
            return "No file part"

        file = request.files["image"]

        # If no file is selected, return an error message
        if file.filename == "":
            return "No selected file"

        # If file is selected, save it and perform classification
        if file:
            upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_image_path)
            label, prob = classify(cnn_model, upload_image_path)
            prob = round((prob * 100), 2)
            return render_template("classify.html", image_file_name=file.filename, label=label, prob=prob)
        else:
            return "No file uploaded"

# Serve uploaded images
@app.route("/uploads/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Display model page
@app.route('/model')
def model():
    return render_template('model.html')

if __name__ == '__main__':
    app.run(debug=True)
