# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F8WAJ-CjYRVFnY0zyuLJL6zcVtb25hnh
"""


from fastai.vision.all import *
from PIL import Image
import streamlit as st

# Load the model
@st.cache_resource
def load_model():
    return load_learner('https://github.com/jk-vishwanath/DAV-6150/blob/main/model.pkl')

model = load_model()

# Streamlit UI
st.title("Eagle or Crow Classifier")
st.write("Upload an image to classify as an Eagle or a Crow.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = PILImage.create(uploaded_file)
    st.image(image.to_thumb(256, 256), caption="Uploaded Image", use_column_width=True)

    # Classify image
    pred_class, pred_idx, probs = model.predict(image)
    st.write(f"Prediction: {pred_class}")
    st.write(f"Confidence: {probs[pred_idx] * 100:.2f}%")

