import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os


model = tf.keras.models.load_model('my_cnn_model_7.h5')  


def predict_image(img):
   
    img = img.resize((64, 64))  
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    
    
    predictions = model.predict(img_array)
    
    prediction_label = (predictions > 0.5).astype("int32")
    
    return prediction_label[0][0]  

# Streamlit UI
st.title("Image Classifier: Real vs Fake")
st.write("Upload an image to classify it as 'Real' or 'Fake'.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    img = image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    
    
    if st.button('Classify'):
        prediction = predict_image(img)
        if prediction == 1:
            st.write("Prediction: The given image is **Real**")
        else:
            st.write("Prediction: The given image is **Fake**")
