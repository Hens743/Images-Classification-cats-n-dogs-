import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

@st.cache_resource
def custom_model():
    try:
        model = tf.keras.models.load_model('data/pruned_model_20231122-201315.keras')  # Model path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = custom_model()

class_names = ['Cat', 'Dog']  # Model's classes

st.title('Image Classification Web App')

uploaded_image = st.file_uploader('Upload an image of dog or cat', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    try:
        image = Image.open(uploaded_image)
        image = image.resize((100, 100))  
        image = np.array(image)
        image = image / 255.0  # Normalize

        if model:
            predictions = model.predict(np.expand_dims(image, axis=0))
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            confidence_score = np.max(predictions) * 100  # Confidence score in percentage

            st.write(f'Prediction: {predicted_class} (Confidence: {confidence_score:.2f}%)')
    except Exception as e:
        st.error(f"Error processing image: {e}")




