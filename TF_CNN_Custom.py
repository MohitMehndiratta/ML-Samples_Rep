from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D
import pandas as pd
import numpy as np
from tensorflow import keras as kr

img_rows=28
img_cols = 28
img_classes = 10


def prep_data(raw):
    y = raw[1:, 0]
    out_y = kr.utils.to_categorical(y)

    x = raw[1:, 1:]
    num_shape = x.shape[0]

    out_x = x.reshape(num_shape, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y


raw_train = pd.read_csv(r'C:\Users\Mohit\Desktop\Sample Files\2243_9243_bundle_archive\fashion-mnist_train.csv')
fashion_file = np.array(raw_train)
x_train, y_train = prep_data(fashion_file)

fashion_model = Sequential()

fashion_model.add(Conv2D(filters=12, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
fashion_model.add(Conv2D(filters=20, kernel_size=3, activation='relu'))
fashion_model.add(Flatten())

fashion_model.add(Dense(100, activation='relu'))
fashion_model.add(Dense(10, activation='softmax'))

fashion_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

fashion_model.fit(x_train,y_train,batch_size=1000,validation_split=0.2)

test_data = pd.read_csv(r'C:\Users\Mohit\Desktop\Sample Files\2243_9243_bundle_archive\fashion-mnist_test.csv')
fashion_test_data=np.array(test_data)
x_test,y_test=prep_data(fashion_test_data)


fashion_model.evaluate(x_test,y_test,verbose=1)