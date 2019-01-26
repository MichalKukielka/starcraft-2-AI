<<<<<<< HEAD

=======
>>>>>>> 2f206275ad1b88bb73a921568ec5b47cbc4aeeed
import keras
import keras.models import Sequential
import keras.layers Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

import os
import numpy as numpy
import random


model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape(176, 200, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

#   fully-connected dense layer
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))

#   output layer
model.add(Dense(4, activation = 'softmax'))

#   compile settings
learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

#   log via tensorboad
tensorboard = TensorBoard(log_dir="logs/stage1")