import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained models with error handling
try:
    breast_cancer_model = tf.keras.models.load_model(r'best_model.keras')  
    brain_tumor_model = tf.keras.models.load_model(r'brain_best.keras') 
    skin_cancer_model = tf.keras.models.load_model(r'skin_best (1).keras')  
except Exception as e:
    st.error(f"Error loading models: {e}")

# Class names for skin cancer
skin_cancer_classes = {
    0: 'Melanoma',
    1: 'Basal Cell Carcinoma',
    2: 'Nevus'
}

# Title and catchy phrase
st.set_page_config(page_title='Early Stage Cancer Classification', layout='wide')
st.title('🩺 Early Stage Cancer Classification')
st.write("**“Detect it early, treat it right.”**")  # Catchy phrase
st.write("Choose the type of cancer classification you want to perform.")

# Sidebar for user input
st.sidebar.header("User Input")
classification_type = st.sidebar.selectbox("Select Classification Type:", ("Breast Cancer", "Brain Tumor", "Skin Cancer"))
uploaded_file = st.sidebar.file_uploader("Upload an image for classification...", type=["jpg", "png"])

# Main area
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
        # Display the image with a reduced width
        st.image(image, caption="Uploaded scan", width=300)  # Set width to 300 pixels
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)

        if classification_type == "Breast Cancer":
            with st.spinner("Classifying for breast cancer..."):
                breast_cancer_prediction = breast_cancer_model.predict(img_array)[0][0]
                breast_confidence = breast_cancer_prediction * 100
                
                if breast_cancer_prediction > 0.5:
                    st.success(f"Likely Malignant: {breast_confidence:.2f}%")
                else:
                    st.success(f"Likely Benign: {100 - breast_confidence:.2f}%")

        elif classification_type == "Brain Tumor":
            with st.spinner("Classifying for brain tumor..."):
                brain_tumor_prediction = brain_tumor_model.predict(img_array)[0][0]
                brain_confidence = brain_tumor_prediction * 100
                
                if brain_tumor_prediction > 0.6:
                    st.success(f"Brain Tumor detected with confidence: {brain_confidence:.2f}%")
                else:
                    st.success(f"No Brain Tumor detected with confidence: {100 - brain_confidence:.2f}%")

        elif classification_type == "Skin Cancer":
            with st.spinner("Classifying for skin cancer..."):
                skin_cancer_prediction = skin_cancer_model.predict(img_array)[0]
                skin_predicted_class = np.argmax(skin_cancer_prediction)
                skin_confidence = np.max(skin_cancer_prediction) * 100
                
                if skin_confidence > 50:  # Assuming > 50% confidence means cancer is detected
                    predicted_class_name = skin_cancer_classes.get(skin_predicted_class, "Unknown")
                    st.success(f"Skin Cancer detected: **{predicted_class_name}** with confidence: {skin_confidence:.2f}%")
                else:
                    st.success("No Skin Cancer detected with high confidence.")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Footer
st.markdown("---")
st.write("### Remember: Early detection can save lives!")
