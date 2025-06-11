import streamlit as st
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

if 'model' not in st.session_state:
    st.session_state['model'] = tf.keras.models.load_model('streamlit-app/number_predictor.keras')

st.title('Number Prediction')

def convert_to_black_and_white_and_normalize(np_array):
    max_val = np_array.max()
    threshold = max_val // 2
    # convert to only black and white
    b_w = np.where(np_array > threshold, 255, 0)

    # convert all 255 -> 1
    normalized = np.where(b_w == 255, 1, 0)
    return normalized

canvas_result = st_canvas(
    fill_color="#000000",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color='#000000',
    background_color="#FFFFFF",
    update_streamlit=True,
    height=112,
    width=224,
    drawing_mode='freedraw',
    point_display_radius=3,
    key="canvas",
)
if canvas_result.image_data is not None:
    image = canvas_result.image_data
    # image = cv2.imread(image)  # Replace 'path_to_image.jpg' with your image file path
    image = cv2.resize(image, (56, 28), interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1, 28, 56, 1)
    image = convert_to_black_and_white_and_normalize(image)
    image = ~image +2


    pred = st.session_state['model'].predict(image)

    st.write(f'Predicted Number: {np.argmax(pred, axis=1)[0]}')
