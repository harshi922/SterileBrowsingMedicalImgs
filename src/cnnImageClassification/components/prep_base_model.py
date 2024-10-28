import os
import urllib.request as request
import zipfile as ZipFile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from cnnImageClassification.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def define_base_model(self):
        self.model = Sequential([
                Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.config.params_image_size[0], self.config.params_image_size[1], self.config.params_image_size[2])),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Flatten(),
                Dense(128, activation='relu'),  # Fully connected layer
                Dropout(0.5),
                
            ])

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, loss, epochs, metrics, batch_size):
      
        prediction = Dense(units=classes, activation="softmax")(model.output)
        full_model = tf.keras.Model(inputs=model.input, outputs=prediction)

        # Compile the model
        full_model.compile(
            optimizer=Adam(),
            loss=loss,
            metrics=metrics
        )

        full_model.summary()
        return full_model
    

    def create_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            loss=self.config.params_loss,
            epochs=self.config.params_epochs,
            metrics=self.config.params_metrics,
            batch_size=self.config.params_batch_size
        )

        self.save_model(path=self.config.base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)