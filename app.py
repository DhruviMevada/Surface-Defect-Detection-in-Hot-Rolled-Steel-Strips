import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load model and labels
model = tf.keras.models.load_model("steel_defect_model.h5")
with open("labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Steel Defect Detection", layout="centered")
st.title("🔍 Surface Defect Detection in Hot-Rolled Steel Strips")
st.markdown(
    """
    Upload a steel strip surface image to detect defects using a trained CNN model.  
    The model can classify **six defect types**:
    - Crazing
    - Inclusion
    - Patches
    - Pitted
    - Rolled-in Scale
    - Scratches
    """
)

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        # Preprocess image
        image = image.resize((200, 200))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)
        pred_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence < 0.60:
            st.warning("⚠️ Low confidence. This image may not contain a known defect.")
        else:
            st.success(f"🧠 **Predicted Defect:** {class_labels[pred_index]}")
            st.info(f"📊 **Confidence:** {confidence*100:.2f}%")

        # Optional: Show full class-wise probabilities
        if st.checkbox("Show all class probabilities"):
            prob_dict = {class_labels[i]: f"{prob*100:.2f}%" for i, prob in enumerate(prediction[0])}
            st.json(prob_dict)

    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")
