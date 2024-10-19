import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
try:
    breast_cancer_model = tf.keras.models.load_model(r'best_model.keras')  
    brain_tumor_model = tf.keras.models.load_model(r'brain_best.keras') 
    skin_cancer_mode=tf.keras.models.load_model(r'skin_best (1).keras')
except Exception as e:
    st.error(f"Error loading models: {e}")
st.title('Early Stage Cancer Classification')
st.write("Choose the type of cancer classification you want to perform.")
classification_type = st.selectbox("Select Classification Type:", ("Breast Cancer", "Brain Tumor","Skin Cancer"))

# Upload image 
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded scan", use_column_width=True)
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  
        if classification_type == "Breast Cancer":
            with st.spinner("Classifying for breast cancer..."):
                breast_cancer_prediction = breast_cancer_model.predict(img_array)[0][0]
                breast_confidence = breast_cancer_prediction * 100
                
                if breast_cancer_prediction > 0.5:
                    st.write(f"Likely Malignant: {breast_confidence:.2f}%")
                else:
                    st.write(f"Likely benign: {100 - breast_confidence:.2f}%")
        elif classification_type == "Brain Tumor":
            with st.spinner("Classifying for brain tumor..."):
                brain_tumor_prediction = brain_tumor_model.predict(img_array)[0][0]
                brain_confidence = brain_tumor_prediction * 100
                
                if brain_tumor_prediction > 0.6:
                    st.write(f"Brain Tumor detected with confidence: {brain_confidence:.2f}%")
                else:
                    st.write(f"No Brain Tumor detected with confidence: {100 - brain_confidence:.2f}%")
                
    except Exception as e:
        st.error(f"Error processing the image: {e}")

