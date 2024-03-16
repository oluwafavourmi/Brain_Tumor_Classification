import streamlit as st
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import img_to_array 
import numpy as np
import pandas as pd

final_model = load_model('brain_tumor.h5')

def predict_function(image, final_model):
    image_array = img_to_array(image)
    image_dim = np.expand_dims(image_array, axis=0)
    predict = final_model.predict(image_dim)
    return predict

image_size = 200

st.title('Brain Tumor Web App')
file_image = st.file_uploader('Upload your Image', type=['jpeg', 'jpg', 'png', 'gif'])
if file_image is None:
    st.write('No file is uploaded here')
else:
    image = image.load_img(file_image, target_size=(image_size, image_size))
    st.image(image, caption='uploaded image', use_column_width=True)
    predictions = predict_function(image, final_model)
    string = ''
    class_names = ['BENING', 'MALIGNANT', 'NORMAL']

    string = 'This is likely the picture of' + '' + class_names[np.argmax(predictions)]

    st.success(predictions)
    st.success(string)
