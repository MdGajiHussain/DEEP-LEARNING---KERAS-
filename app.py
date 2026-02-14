import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM CSS (Dark Look) ----------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .stFileUploader {
            background-color: #161b22;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🧬 Skin Cancer Detection Application")

st.write(
    "This is a skin cancer detection application. Upload an image, and the model "
    "will predict whether the skin lesion is **Malignant** or **Benign**."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model("skin_cancer_cnn.h5")  # or .keras

model = load_cnn_model()

# ---------------- IMAGE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    class_names = ["Benign", "Malignant"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown("---")
    st.subheader("🧪 Prediction Result")

    if predicted_class == "Malignant":
        st.error(f"🛑 **{predicted_class}**")
    else:
        st.success(f"✅ **{predicted_class}**")

    st.info(f"Confidence: **{confidence:.2f}%**")

# ---------------- ABOUT SECTION ----------------
st.markdown("---")
st.subheader("📌 About the Model")

st.write(
    "This model uses a **CNN (Convolutional Neural Network)** architecture for "
    "predicting whether a skin lesion is **Benign** or **Malignant** based on "
    "images of skin lesions."
)

# ---------------- FEATURES ----------------
st.subheader("✨ Features")

st.markdown("""
- **Input:** Skin lesion images  
- **Output:** Benign or Malignant classification  
- **Model:** CNN trained on skin cancer dataset  
- **Frameworks:** TensorFlow, Keras, Streamlit  
""")
