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