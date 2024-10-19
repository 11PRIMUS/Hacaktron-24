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
        
        # Breast Cancer Prediction (Binary Classification)
        with st.spinner("Classifying for breast cancer..."):
            breast_cancer_prediction = breast_cancer_model.predict(img_array)[0][0]
            breast_confidence = breast_cancer_prediction * 100
            if breast_cancer_prediction > 0.5:
                st.write(f"Breast Cancer detected with confidence: {breast_confidence:.2f}%")
                st.success("Stopping further checks as cancer was detected.")
                st.stop()  # Stop further checks

            st.write(f"No Breast Cancer detected with confidence: {100 - breast_confidence:.2f}%")

        # Brain Tumor Prediction (Binary Classification)
        with st.spinner("Classifying for brain tumor..."):
            brain_tumor_prediction = brain_tumor_model.predict(img_array)[0][0]
            brain_confidence = brain_tumor_prediction * 100
            if brain_tumor_prediction > 0.5:
                st.write(f"Brain Tumor detected with confidence: {brain_confidence:.2f}%")
            else:
                st.write(f"No Brain Tumor detected with confidence: {100 - brain_confidence:.2f}%")
            st.stop()
    except Exceptiom as e:
        st.error(f"error processing image:{e}")
