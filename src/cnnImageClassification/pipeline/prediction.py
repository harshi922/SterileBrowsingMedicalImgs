import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2


class PredictionPipeline:
    def __init__(self):
        model_path = os.path.join("artifacts","training", "model.h5")
        self.model = load_model(model_path)
        
    def predict(self, fileimg):
            # load model

            image = cv2.imread(fileimg)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, [100, 120])
            gray_image = gray_image.reshape(1, 100, 120, 1)
            predicted_class = np.argmax(self.model.predict(gray_image))

            if predicted_class == 0:
                prediction = "Blank"
            elif predicted_class == 1:
                prediction =  "OK"
            elif predicted_class == 2:
                prediction =  "Thumbs Up"
            elif predicted_class == 3:
                prediction =  "Thumbs Down"
            elif predicted_class == 4:
                prediction =  "Punch"
            elif predicted_class == 5:
                prediction = "High Five"
            return prediction