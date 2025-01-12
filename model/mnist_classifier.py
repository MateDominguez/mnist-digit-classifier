import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class MNISTClassifier:
    def _init_(self):
        self.model = None
        self.history = None       

    def build_model(self, dropout_rate=0.3):
        '''
        CNN model using Functional API for better layer visibility.
        This aproach explicitly defines the imput shapes and connections between layers.
        '''
        # Define the input layer
        inputs = tf.keras.Input(shape=(28, 28, 1), name='input_layer')

        # First Convolutional Block
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.Activation('relu', name='relu1')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='dropout1')(x)

        # Second Convolutional Block
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn2')(x)
        x = tf.keras.layers.Activation('relu',name='relu2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='dropout2')(x)

        # Third Convolutional Block
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv3')(x)
        x = tf.keras.layers.BatchNormalization(name='bn3')(x)
        x = tf.keras.layers.Activation('relu',name='relu3')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='dropout3')(x)

        # Dense layers
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense1')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='dropout4')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax', name='outputs')(x)

        # Create the model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_classifier')

        return self


    def compile_model(self, learning_rate=0.001):
        '''Compile the model with given learning rate.'''
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        '''Train the model and save training history.'''
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        return self