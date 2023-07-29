import streamlit as st
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2

# Function to load a model based on the user's selection
def load_selected_model(model_name):
    if model_name == "model_xception":
        return load_model("model_xception.h5")
    elif model_name == "model_DenseNet121":
        return load_model("model_DenseNet121.h5")
    elif model_name == "model_InceptionV3":
        return load_model("model_InceptionV3.h5")
    elif model_name == "model_kaggle_alzheimer":
        return load_model("model_kaggle_alzheimer.h5")
    # Add more models here if needed

# Set app title and header
st.set_page_config(page_title="Alzheimer's Disease Detection",
                   page_icon=":microscope:", layout="wide")
st.title("Alzheimer's Disease Detection")

# Add an image to the app
image = Image.open('image_presentation.jpg')
st.image(image, caption='Brain MRI scans')

# Add a subheader and description
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the patient's MRI image.")
st.write("This application uses a convolutional neural network (CNN) model.")

# Add a dropdown menu to select the model
selected_model = st.selectbox("Select a model", ["model_xception", "model_DenseNet121" ,"model_InceptionV3","model_kaggle_alzheimer" ])

# Load the selected model
model = load_selected_model(selected_model)

# Add a file uploader to the app
file = st.file_uploader("Please upload an MRI image.", type=["jpg", "png"])

# Define a function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (176, 208)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Make predictions on uploaded images
if file is None:
    st.text("No image file has been uploaded.")
else:
    image = Image.open(file)
    predictions = import_and_predict(image, model)
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    string = "The patient is predicted to be: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.image(image)
