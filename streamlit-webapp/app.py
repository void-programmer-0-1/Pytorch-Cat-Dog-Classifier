import streamlit as st
import numpy as np
import cv2
from prediction import predict

st.set_page_config(page_title='Pet Classifier')
st.write("""
### Pet Animals Classifier
""")

image = None
classes = {0:"cat",1:"dog"}

file = st.file_uploader("pick a file")
if file:
    image_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, 1)
    st.image(image, channels="BGR",width=400)

if st.button('Predict'):
    predited = predict(image)
    st.write(classes[predited])