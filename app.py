import streamlit as st
import requests
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import os

# Must be first Streamlit command
st.set_page_config(page_title="Osteoarthritis", layout="centered")

# Define label dictionary
dic = {0: 'Grade 0 : Normal', 1: 'Grade 1 : Doubtful', 2: 'Grade 2 : Mild', 3: 'Grade 3 : Moderate', 4: 'Grade 4 : Severe'}

# Image size
img_size = 256

# Download the model if not already downloaded
MODEL_URL = "https://huggingface.co/SPPramod/model2.keras/resolve/main/model2.keras"
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)

# Load model
model = load_model(MODEL_PATH)

# Define predict function
def predict_label(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)
    p = np.argmax(model.predict(i), axis=-1)
    return dic[p[0]]

# Streamlit UI
st.title("Diagnosis for the Prediction of Knee Osteoarthritis Using Deep Learning")
st.write("Choose your Knee X-Ray file and click Predict to get your diagnosis.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        prediction_text = predict_label(img)
        st.success(f"Prediction: **{prediction_text}**")
