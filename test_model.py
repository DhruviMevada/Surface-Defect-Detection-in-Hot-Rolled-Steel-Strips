import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model("steel_defect_model.h5")

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Test image path
test_dir = "test_images"  # Folder containing test JPG/PNG images

# Make sure test_images directory exists
if not os.path.exists(test_dir):
    print(f"❌ Folder '{test_dir}' not found. Please create it and add images to test.")
    exit()

# Loop through each image in the test directory
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_dir, filename)
        print(f"\n📷 Testing: {filename}")

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((200, 200))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        pred_class = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"✅ Prediction: {pred_class} ({confidence:.2f}%)")
