import os
import io
import cv2
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Config
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "runs/detect"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model once (IMPORTANT)
model = YOLO("best.pt")


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":

        # Check if file is uploaded
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        # Secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save file
        file.save(filepath)

        # Read image using OpenCV
        img = cv2.imread(filepath)

        if img is None:
            return "Invalid image file"

        # Convert to PIL Image
        _, buffer = cv2.imencode(".jpg", img)
        image = Image.open(io.BytesIO(buffer.tobytes()))

        # Run YOLO prediction
        results = model.predict(image, save=True)

        # Get latest result folder
        subfolders = [
            f for f in os.listdir(RESULT_FOLDER)
            if os.path.isdir(os.path.join(RESULT_FOLDER, f))
        ]

        if not subfolders:
            return "No detection results found"

        latest_folder = max(
            subfolders,
            key=lambda x: os.path.getctime(os.path.join(RESULT_FOLDER, x))
        )

        result_dir = os.path.join(RESULT_FOLDER, latest_folder)

        # Get output image
        result_files = os.listdir(result_dir)

        if not result_files:
            return "No output image generated"

        result_image = result_files[0]

        # Return detected image
        return send_from_directory(result_dir, result_image)

    return render_template("index.html")


# Optional route to access result images directly
@app.route("/results/<path:filename>")
def display_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)