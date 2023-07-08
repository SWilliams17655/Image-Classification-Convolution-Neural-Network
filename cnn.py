import keras_preprocessing.image
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.style as style


class CNN:

    def __init__(self, class_names):
        """
        Instantiates CNN model for use in training an image classifier.
        :return: A summary of the CNN model developed.
        """
        self.class_names = class_names
        self.cnn_model = keras.Sequential()
        self.training_history = []

        self.cnn_model.add(keras.layers.Conv2D(filters=32,
                                               kernel_size=7,
                                               activation='relu',
                                               padding='same',
                                               input_shape=[128, 128, 3]))
        self.cnn_model.add(keras.layers.Activation('relu'))
        self.cnn_model.add(keras.layers.MaxPool2D())

        self.cnn_model.add(keras.layers.Conv2D(filters=32,
                                               kernel_size=5,
                                               activation='relu',
                                               padding='same',
                                               input_shape=[128, 128, 3]))
        self.cnn_model.add(keras.layers.Activation('relu'))
        self.cnn_model.add(keras.layers.MaxPool2D())

        self.cnn_model.add(keras.layers.Conv2D(filters=32,
                                               kernel_size=3,
                                               activation='relu',
                                               padding='same',
                                               input_shape=[128, 128, 3]))
        self.cnn_model.add(keras.layers.Activation('relu'))
        self.cnn_model.add(keras.layers.MaxPool2D())

        self.cnn_model.add(keras.layers.Conv2D(filters=32,
                                               kernel_size=3,
                                               activation='relu',
                                               padding='same',
                                               input_shape=[128, 128, 3]))
        self.cnn_model.add(keras.layers.Activation('relu'))
        self.cnn_model.add(keras.layers.MaxPool2D())

        self.cnn_model.add(keras.layers.Flatten())
        self.cnn_model.add(keras.layers.Dense(64))
        self.cnn_model.add(keras.layers.Activation('relu'))
        self.cnn_model.add(keras.layers.Dropout(0.2))
        self.cnn_model.add(keras.layers.Dense(32))
        self.cnn_model.add(keras.layers.Activation('relu'))
        self.cnn_model.add(keras.layers.Dropout(0.2))
        self.cnn_model.add(keras.layers.Dense(5))
        self.cnn_model.add(keras.layers.Activation('sigmoid'))

        self.cnn_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def train_model(self, training_file_location, validation_file_location, epochs=10):
        """
        Trains convolution neural network to classify a three channel image with dimensions 128 x 128.
        :param training_file_location : String with relative file location of training folder.
        :param validation_file_location : String with relative file location of validation data folder.
        :return: Returns the graph summary of the CNN model.
        """

        train_image_set = keras.preprocessing.image_dataset_from_directory(
            training_file_location,
            labels="inferred",
            label_mode='categorical',
            image_size=[128, 128],
            class_names=self.class_names,
            interpolation='nearest',
            color_mode='rgb',
            batch_size=64,
            shuffle=True,
        )
        validate_image_set = keras.preprocessing.image_dataset_from_directory(
            validation_file_location,
            labels="inferred",
            label_mode='categorical',
            image_size=[128, 128],
            class_names=self.class_names,
            interpolation='nearest',
            color_mode='rgb',
            batch_size=64,
            shuffle=True,
        )

        self.training_history = self.cnn_model.fit(train_image_set,
                                                   epochs=epochs,
                                                   validation_data=validate_image_set,
                                                   )

        return self.cnn_model.summary()

    def graph_training_history(self):
        """
        Returns a line graph representing the historical results of the model training effort.
        :return: A line graph representing the accuracy of each CNN training epoch.
        """
        style.use('dark_background')

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
        # summarize history for accuracy
        ax1.plot(self.training_history.history['accuracy'])
        ax1.plot(self.training_history.history['val_accuracy'])
        ax1.set_title('CNN Model Accuracy', fontsize=20)
        ax1.set_ylabel('Accuracy', fontsize=18)
        ax1.set_yticks(np.arange(0.0, 1.0, .1))
        ax1.tick_params(axis='both', labelsize=16)
        ax1.set_xlabel('Epoch', fontsize=18)
        ax1.legend(['Train', 'Validate'], loc='upper left', fontsize=16)
        fig.show()
        return fig

    def classify_image(self, image):
        """
        Takes an image as input following a model being trained then classifies that image.
        :param image : Image to be classified.
        :return: Returns the array location and classifier name of the classification for that image.
        """

        img_array = keras_preprocessing.image.img_to_array(image)
        img_batch = np.expand_dims(img_array, axis=0)
        pred = self.cnn_model.predict(img_batch)
        array_loc = np.argmax(pred)
        class_label = self.class_names[np.argmax(pred)]
        print(f"This image was classified as a {class_label} which is located in position {array_loc}")
        return array_loc, class_label
