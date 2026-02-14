
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# ------------------ CONFIG ------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ APP INIT ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ MODEL -------------------
model = load_model("facial_emotion_detection_model.h5")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ------------------ UTILS -------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_emotion(img_path):
    img = image.load_img(
        img_path,
        target_size=(48, 48),
        color_mode="grayscale"
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    idx = np.argmax(prediction)

    return class_names[idx], round(float(prediction[0][idx]) * 100, 2)

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            emotion, confidence = detect_emotion(file_path)

            return render_template(
                "index.html",
                image_path=f"static/uploads/{filename}",
                emotion=emotion,
                confidence=confidence
            )

        return render_template("index.html", error="Invalid file format")

    return render_template("index.html")

# ------------------ RUN ---------------------
if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model
# model = load_model('facial_emotion_detection_model.h5')

# # Define class names
# class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# # Upload folder
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Emotion detection function
# def detect_emotion(img_path):
#     img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_index = np.argmax(prediction)
#     predicted_class = class_names[predicted_index]
#     confidence = round(prediction[0][predicted_index] * 100, 2)

#     return predicted_class, confidence

# # Home route
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return 'No file uploaded!'
#         file = request.files['file']
#         if file.filename == '':
#             return 'No file selected!'

#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)

#             # Detect emotion
#             emotion, confidence = detect_emotion(file_path)

#             return render_template('index.html', image_path=file_path, emotion=emotion, confidence=confidence)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)




# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # Page config
# st.set_page_config(page_title="Facial Emotion Detection", layout="centered")

# # Load model (cache for performance)
# @st.cache_resource
# def load_emotion_model():
#     return load_model("facial_emotion_detection_model.h5")

# model = load_emotion_model()

# # Class names
# class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# # Emotion detection function
# def detect_emotion(img):
#     img = img.convert("L")              # Convert to grayscale
#     img = img.resize((48, 48))

#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array, verbose=0)
#     predicted_index = np.argmax(prediction)

#     predicted_class = class_names[predicted_index]
#     confidence = round(float(prediction[0][predicted_index]) * 100, 2)

#     return predicted_class, confidence

# # UI
# st.title("😊 Facial Emotion Detection")
# st.write("Upload a face image to detect emotion")

# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     img = Image.open(uploaded_file)

#     st.image(img, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Detecting emotion..."):
#         emotion, confidence = detect_emotion(img)

#     st.success(f"**Emotion:** {emotion.upper()}")
#     st.info(f"**Confidence:** {confidence}%")
