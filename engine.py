import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('data/pruned_model.keras')

model = load_model()
class_names = ['Cat', 'Dog']  # Model's classes

# Set up the Streamlit app
st.title('Image Classification Web App')
uploaded_image = st.file_uploader('Upload an image of a dog or cat', type=['jpg', 'png', 'jpeg'])

# Function to make a prediction
def make_prediction(image_data):
    image_resized = Image.open(image_data).resize((100, 100))
    image_array = np.array(image_resized) / 255.0  # Normalize
    predictions = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class_index = np.argmax(predictions)
    return class_names[predicted_class_index], np.max(predictions) * 100

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    try:
        predicted_class, confidence_score = make_prediction(uploaded_image)
        st.write(f'Prediction: {predicted_class} (Confidence: {confidence_score:.2f}%)')
    except Exception as e:
        st.error(f"Error processing the image: {e}")


