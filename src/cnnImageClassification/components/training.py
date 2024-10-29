from cnnImageClassification.entity.config_entity import TrainingConfig
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )
    
    def prepare_training_testing_data(self):

        # assume this is the path to the dataset

        dataset_root = self.config.training_data

        max_images_per_gesture = 1600
        loaded_images = []
        output_vectors = []

        # List of gesture folders (0 to 5)
        
        list_of_gestures = ['blank', 'ok', 'thumbsup', 'thumbsdown', 'fist', 'five']

        # Load images and create one-hot encoded labels
        for gesture in list_of_gestures:
            # Construct the path to the current gesture folder
            gesture_path = os.path.join(dataset_root, gesture, '*')
            gest_image_paths = glob.glob(gesture_path)
            
            # Load images and resize them
            for i, image_path in enumerate(gest_image_paths):
                if i < max_images_per_gesture:
                    image = cv2.imread(image_path)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_image = cv2.resize(gray_image, (100, 120))
                    loaded_images.append(gray_image)
                    
                    # Create the one-hot encoded output vector
                    one_hot_vector = [0] * len(list_of_gestures)  # Initialize vector with zeros
                    gesture_index = list_of_gestures.index(gesture)
                    one_hot_vector[gesture_index] = 1  # Set the index corresponding to the current gesture to 1
                    output_vectors.append(one_hot_vector)
        self.X = np.array(loaded_images)
        self.y = np.array(output_vectors)
        print(self.X.shape)
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=4
        )

        # Reshape the data
        self.X_train = self.X_train.reshape(
            self.X_train.shape[0],
            self.config.params_image_size[0],
            self.config.params_image_size[1],
            self.config.params_image_size[2]
        )
        self.X_test = self.X_test.reshape(
            self.X_test.shape[0],
            self.config.params_image_size[0],
            self.config.params_image_size[1],
            self.config.params_image_size[2]
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):
       
        self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.config.params_batch_size,
            epochs=self.config.params_epochs,
            verbose=1,
            validation_data=(self.X_test, self.y_test),
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )