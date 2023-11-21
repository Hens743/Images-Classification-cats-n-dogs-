import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import tempfile
import os

# Function to load a Keras model from an uploaded file
def load_model(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        model = keras.models.load_model(tmp_path)

        # Optionally, remove the temporary file if desired
        os.remove(tmp_path)

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

st.title('Image classification web app')

model_upload = st.file_uploader('Upload a model file', type=['h5', 'keras'])

if model_upload is not None:
    model = load_model(model_upload)
    if model:
        st.success("Model loaded successfully!")

        uploaded_image = st.file_uploader('Upload an image of a DOG or CAT', type=['jpg', 'png', 'jpeg'])

        if uploaded_image is not None:
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

            try:
                image = Image.open(uploaded_image)
                image = image.resize((100, 100))  # Adjust size as needed
                image = np.array(image)
                image = image / 255.0  # Normalize

                predictions = model.predict(np.expand_dims(image, axis=0))
                class_names = ['CAT', 'DOG']  # Model's classes
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_names[predicted_class_index]
                confidence_score = np.max(predictions) * 100  # Confidence score in percentage

                st.write(f'Prediction: {predicted_class} (Confidence: {confidence_score:.2f}%)')
            except Exception as e:
                st.error(f"Error processing image: {e}")
