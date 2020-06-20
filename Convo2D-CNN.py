import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameter

fashion_mnist = keras.datasets.fashion_mnist

(data_set_train, data_set_train_label), (data_set_test, data_set_test_label) = fashion_mnist.load_data()

train_images = data_set_train / 255.0
test_images = data_set_test / 255.0

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)


def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=hp.Int('Conv_1_filter', min_value=32, max_value=128, step=16),
                            kernel_size=hp.Int('Conv_1_Kernel', 3,5), activation='relu'
                            , input_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=hp.Int('Conv_2_filter', min_value=32, max_value=128, step=16),
                            kernel_size=hp.Int('Conv_2_Kernel', 3,5), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=hp.Int('dense_units_1', min_value=30, max_value=128, step=16), activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 0.1, 0.2)),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


tuner_search = RandomSearch(build_model, objective='val_accuracy',
                            max_trials=5, executions_per_trial=1, directory='C:/', project_name="mnist_fashion")
tuner_search.search(train_images, data_set_train_label, epochs=3, validation_split=0.1,batch_size=1000)