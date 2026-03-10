import os
import json
import numpy as np
import wikipedia
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================
# LOAD TRAINED MODEL
# =========================
model = load_model("flower_model.h5")


# =========================
# LOAD CLASS LABELS
# =========================
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# Convert dictionary keys to list
class_names = list(class_indices.keys())


# =========================
# WIKIPEDIA FUNCTION
# =========================
def get_flower_info(name):
    try:
        wikipedia.set_lang("en")

        search_query = name + " flower"

        summary = wikipedia.summary(search_query, sentences=3)
        page = wikipedia.page(search_query)

        return summary, page.url

    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            page = wikipedia.page(e.options[0])
            return summary, page.url
        except:
            return "No clear information found.", None

    except:
        return "No information found on Wikipedia.", None


# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]

        if file and file.filename != "":

            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # =========================
            # IMAGE PREPROCESSING
            # =========================
            img = image.load_img(filepath, target_size=(64, 64))
            img = image.img_to_array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # =========================
            # MODEL PREDICTION
            # =========================
            result = model.predict(img)

            predicted_index = np.argmax(result)
            confidence = round(np.max(result) * 100, 2)

            predicted_flower = class_names[predicted_index]

            # =========================
            # WIKIPEDIA INFO
            # =========================
            summary, url = get_flower_info(predicted_flower)

            return render_template(
                "index.html",
                prediction=predicted_flower.capitalize(),
                confidence=confidence,
                summary=summary,
                url=url,
                image_path=filepath
            )

    return render_template("index.html")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)