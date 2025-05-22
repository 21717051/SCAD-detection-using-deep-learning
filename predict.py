
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

model = load_model('./models/cnn_model.h5')

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return "SCAD" if prediction > 0.5 else "Normal"
