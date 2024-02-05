import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from keras.layers import Conv2D, UpSampling2D, InputLayer, Input
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.preprocessing.image import ImageDataGenerator
import os
import urllib.request
from time import sleep

# Download images for training (you can replace this with your dataset)
for i in range(0, 1000):
    url = "https://source.unsplash.com/random"
    filename = f"./data/train/{i}.png"
    urllib.request.urlretrieve(url, filename)
    sleep(1.5)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

path = './data/'
train = train_datagen.flow_from_directory(
    path,
    target_size=(256, 256),
    class_mode=None,
    batch_size=500
)

# Prepare training data
X = []
Y = []

for img in train[0]:
    lab = rgb2lab(img / 255)
    X.append(lab[:, :, 0])
    Y.append(lab[:, :, 1:] / 128)

X = np.array(X).reshape((-1, 256, 256, 1))
Y = np.array(Y).reshape((-1, 256, 256, 2))

# Building the neural network
model = Sequential([
    InputLayer(input_shape=(256, 256, 1)),
    Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(2, (3, 3), activation='tanh', padding='same')
])

# Finish model
optimizer = Adagrad()
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Train the neural network
model.fit(x=X, y=Y, batch_size=1, epochs=1000)
print(model.evaluate(X, Y, batch_size=1))
