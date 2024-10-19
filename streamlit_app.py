import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
try:
    image_type_model = tf.keras.models.load_model(r'image_type_classifier.keras')  # Pre-classifier model
    breast_cancer_model = tf.keras.models.load_model(r'best_model.keras')  
    brain_tumor_model = tf.keras.models.load_model(r'brain_best.keras')  
except Exception as e:
    st.error(f"Error loading models: {e}")

st.title('Early Stage Cancer Classification')
st.write("Automatically classify the image type and perform cancer classification (breast, brain).")
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded scan", use_column_width=True)
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        with st.spinner("Classifying image type..."):
            image_type_prediction = image_type_model.predict(img_array)[0]
            is_breast_scan = image_type_prediction[0] > 0.5  
            
            if is_breast_scan:
                st.write("Image detected as *Breast MRI Scan*.")
                
                # Perform Breast Cancer Prediction (Binary Classification)
                with st.spinner("Classifying for breast cancer..."):
                    breast_cancer_prediction = breast_cancer_model.predict(img_array)[0][0]
                    breast_confidence = breast_cancer_prediction * 100
                    
                    if breast_cancer_prediction > 0.5:
                        st.write(f"Breast Cancer detected with confidence: {breast_confidence:.2f}% (Malignant)")
                    else:
                        st.write(f"No Breast Cancer detected with confidence: {100 - breast_confidence:.2f}% (Benign)")
            
            else:
                st.write("Image detected as *Brain MRI Scan*.")
                
                # Perform Brain Tumor Prediction (Binary Classification)
                with st.spinner("Classifying for brain tumor..."):
                    brain_tumor_prediction = brain_tumor_model.predict(img_array)[0][0]
                    brain_confidence = brain_tumor_prediction * 100
                    
                    if brain_tumor_prediction > 0.6:
                        st.write(f"Brain Tumor detected with confidence: {brain_confidence:.2f}% (Malignant)")
                    else:
                        st.write(f"No Brain Tumor detected with confidence: {100 - brain_confidence:.2f}% (Benign)")
                    
    except Exception as e:
        st.error(f"Error processing the image: {e}")
