import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

loaded_model = tf.keras.models.load_model('lis_image.hdf5')

developerMap = {
    0: 'Bhabuk Kunwar',
    1: 'Bishnu Das',
    2: 'Sunil Shrestha',
    3: 'Suraj Ojha'
}

def import_and_predict(image_data):
    size = (128,128) 
    IMG_SIZE = 128   
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(128, 128),    interpolation=cv2.INTER_CUBIC))/255.
    # print(img_resize.shape)
    
    img_reshape = img_resize.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return np.argmax(loaded_model.predict([img_reshape]))

st.title("LIS Developers Image Classification")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, width=400)
    prediction = import_and_predict(image)
    st.subheader("Developer: " + developerMap[prediction])

        
