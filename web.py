import streamlit as st
from filter import Filter
from PIL import Image
import numpy as np 
import cv2
import io

st.title("Filter Images")

img = st.file_uploader("Upload any image", accept_multiple_files=False)
option = st.selectbox('What would you like to do to your image?',
                       ('', 'Defaut Blur', 'Guassian Blur', 'Detect Vertical Edges', 
                        'Detect Horizontal Edges', 'Sharpen Image'))

if img is not None:
    img_pil = Image.open(img)
    np_img = np.array(img_pil)

    if option == 'Defaut Blur':
        size = st.slider('Blur Intensity', 3, 20)

        fil = Filter(img_pil)
        out = fil.default_blur(np_img.shape[2], size)

        _, buffer = cv2.imencode(".png", out)
        image_file = io.BytesIO(buffer)

        st.download_button("Press to Download!", image_file, file_name='your_image.png', mime='image/png')
        st.title("Preview" )
        st.image(out, channels='BGR')

    elif option == 'Guassian Blur':
        size = st.slider('Blur Intensity', 3, 20)
        sigma = st.slider('Roughness', 1, 1000)

        fil = Filter(img_pil)
        out = fil.guassian_blur(np_img.shape[2], size, sigma)

        _, buffer = cv2.imencode(".png", out)
        image_file = io.BytesIO(buffer)

        st.download_button("Press to Download!", image_file, file_name='your_image.png', mime='image/png')
        st.title("Preview" )
        st.image(out, channels='BGR')


    elif option == 'Detect Horizontal Edges':
        size = st.slider('Detection Range', 3, 20)

        fil = Filter(img_pil)
        out = fil.vert_edge(np_img.shape[2], size)

        _, buffer = cv2.imencode(".png", out)
        image_file = io.BytesIO(buffer)

        st.download_button("Press to Download!", image_file, file_name='your_image.png', mime='image/png')
        st.title("Preview" )
        st.image(out, channels='BGR')


    elif option == 'Detect Vertical Edges':
        size = st.slider('Detection Range', 3, 20)

        fil = Filter(img_pil)
        out = fil.horiz_edge(np_img.shape[2], size)

        _, buffer = cv2.imencode(".png", out)
        image_file = io.BytesIO(buffer)

        st.download_button("Press to Download!", image_file, file_name='your_image.png', mime='image/png')
        st.title("Preview" )
        st.image(out, channels='BGR')


    elif option == 'Sharpen Image':
        size = 3

        fil = Filter(img_pil)
        out = fil.default_blur(np_img.shape[2], size)

        _, buffer = cv2.imencode(".png", out)
        image_file = io.BytesIO(buffer)

        st.download_button("Press to Download!", image_file, file_name='your_image.png', mime='image/png')
        st.title("Preview" )
        st.image(out, channels='BGR')

