from keras.datasets import cifar10
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.constraints import max_norm
# from scipy.misc import toimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
import keras.backend.common as K
from PIL import Image

K.set_image_dim_ordering('th')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

gs = gspec.GridSpec(4, 4, wspace=0.0)
ax = [plt.subplot(gs[i]) for i in range(4 * 4)]
for i in range(16):
    ax[i].imshow(Image.fromarray(x_train[i]))
plt.show()

y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

num_classes = 10
model = Sequential()

# First Convolution layer
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))

# Second Convolution layer
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))

# Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the Output
model.add(Flatten())
model.add(Dense(512, activation='relu'))

# Output Class
model.add(Dense(num_classes, activation='softmax'))

epochs = 50
lrate = 0.05
sgd = SGD(lr=lrate, momentum=0.8, decay=lrate / epochs, nesterov=False)

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Print Summary of CNN
# print(model.summary())

model.fit(x_train, y_train_onehot, validation_data=(x_test, y_test_onehot), epochs=250, batch_size=100)

loss, accuracy = model.evaluate(x_test, y_test_onehot, verbose=0,batch_size=1)
# print('Model Accuracy={:.4f}'.format(accuracy))
print(loss, accuracy)
