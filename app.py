from flask import Flask, render_template, request
import os
from ensemble_model import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Home page
@app.route("/")
def home():
    return render_template("home.html")


# Upload page + prediction
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        prediction, probs, heatmap = predict_image(filepath)

        return render_template(
            "results.html",
            prediction=prediction,
            uploaded_image="uploads/" + file.filename,
            combined_gradcam="uploads/" + os.path.basename(heatmap)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)