from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def sc_v6():
  model = Sequential()
  model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape = (64,64,3)))
  model.add(Activation('relu'))

  model.add(Convolution2D(32, (3, 3)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(64,(3, 3), padding='same'))
  model.add(Activation('relu'))

  model.add(Convolution2D(64, (3, 3)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())

  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(24))
  model.add(Activation('sigmoid'))

  return model
