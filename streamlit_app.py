import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained models

breast_cancer_model = tf.keras.models.load_model(r'hackatron/best_model.keras')
brain_tumor_model = tf.keras.models.load_model(r'hackatron/brain_best.keras')

# Title and instructions
st.title('Cancer Classification (Breast, Brain Tumor)')
st.write("Classify breast cancer, brain tumor using deep learning models.")
uploaded_file = st.file_uploader("Choose an image for cancer classification...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded scan", use_column_width=True)
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
