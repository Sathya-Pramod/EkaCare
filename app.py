import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os

# Must be first Streamlit command
st.set_page_config(page_title="Osteoarthritis", layout="centered")

# Label dictionary
dic = {
    0: 'Grade 0 : Normal',
    1: 'Grade 1 : Doubtful',
    2: 'Grade 2 : Mild',
    3: 'Grade 3 : Moderate',
    4: 'Grade 4 : Severe'
}

img_size = 256

# Model path (STRING, not model)
MODEL_PATH = "model.keras"

# Stop if model file is missing
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

# Cache model (VERY important for Streamlit)
@st.cache_resource
def load_dl_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_dl_model()

# Prediction function
def predict_label(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)
    p = np.argmax(model.predict(i, verbose=0), axis=-1)
    return dic[p[0]]

# UI
st.title("Diagnosis for the Prediction of Knee Osteoarthritis Using Deep Learning")
st.write("Choose your Knee X-Ray file and click Predict to get your diagnosis.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image file.")
        st.stop()

    st.image(
        img,
        channels="BGR",
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):
        prediction_text = predict_label(img)
        st.success(f"Prediction: **{prediction_text}**")
