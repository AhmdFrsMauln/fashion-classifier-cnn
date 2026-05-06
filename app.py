import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("fashion_model.h5")

# Label
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion Item Classifier (CNN)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write("Prediction:", result)
    st.write("Confidence:", confidence)
