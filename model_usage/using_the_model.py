import tensorflow as tf
import cv2
import numpy as np

# load the model
model = tf.keras.models.load_model('../model_usage/number_predictor.keras')

# function to convert image to black and white and normalize
def convert_to_black_and_white_and_normalize(np_array):
    max_val = np_array.max()
    threshold = max_val // 2
    # convert to only black and white
    b_w = np.where(np_array > threshold, 255, 0)

    # convert all 255 -> 1
    normalized = np.where(b_w == 255, 1, 0)
    return normalized

image = cv2.imread('../model_usage/sample_number_35.png')
image = cv2.resize(image, (56, 28), interpolation=cv2.INTER_LINEAR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.reshape(1, 28, 56, 1)
image = convert_to_black_and_white_and_normalize(image)
image = ~image +2


pred = model.predict(image)

print(f'Predicted Number: {np.argmax(pred, axis=1)[0]}')

