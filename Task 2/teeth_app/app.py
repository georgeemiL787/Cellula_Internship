import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
from PIL import Image
import time

# -----------------------
# Configuration
# -----------------------
IMG_SIZE = 256

# MUST match training dataset order
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# -----------------------
# Page Setup
# -----------------------
st.set_page_config(
    page_title="Dental Image Classifier",
    layout="wide"
)

st.title("Dental Condition Classification")
st.write(
    "Upload a dental image to classify potential tooth conditions "
    "using a deep learning model."
)

# -----------------------
# Load Model (Cached)
# -----------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "teeth_transfer_learning_final_v1.keras"
        )
        return model
    except Exception as e:
        st.error("Model loading failed.")
        st.exception(e)
        return None

model = load_model()

# -----------------------
# Preprocessing
# -----------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    # Remove alpha channel if exists
    if image.shape[-1] == 4:
        image = image[..., :3]

    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Instructions")

    st.markdown("""
    1. Upload a dental image  
    2. Wait for prediction  
    3. Review confidence levels  
    """)

    st.markdown("---")
    st.write("Model: EfficientNetB3 Transfer Learning")
    st.write("Input Size: 256 x 256 RGB")

    if model is not None:
        st.write("Model Output Classes:", model.output_shape[-1])
        st.write("CLASS_NAMES Count:", len(CLASS_NAMES))

# -----------------------
# Upload Section
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Dental Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        # Show uploaded image
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)

        with col2:
            if model is None:
                st.stop()

            with st.spinner("Analyzing image..."):

                processed = preprocess_image(image)

                start = time.time()
                preds = model.predict(processed)
                end = time.time()

                probs = preds[0]

                predicted_index = int(np.argmax(probs))

                # Safety check
                if predicted_index < len(CLASS_NAMES):
                    predicted_class = CLASS_NAMES[predicted_index]
                else:
                    predicted_class = "Unknown"

                confidence = probs[predicted_index]

            # -----------------------
            # Prediction Output
            # -----------------------
            st.subheader("Prediction Result")
            st.write(f"Predicted Condition: **{predicted_class}**")
            st.write(f"Confidence: {confidence:.2%}")
            st.caption(f"Inference time: {end-start:.3f} seconds")

            # -----------------------
            # Probability Distribution
            # -----------------------
            st.subheader("Class Probability Distribution")

            prob_dict = dict(zip(CLASS_NAMES, probs))
            st.bar_chart(prob_dict)


    except Exception as e:
        st.error("Image processing failed.")
        st.exception(e)
