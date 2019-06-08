from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

def sc_v6(options):
	model = Sequential()

	if K.image_dim_ordering() == 'th':
		input_shape = (options.img_ch,options.img_rows,options.img_cols)
	else:
		input_shape = (options.img_rows,options.img_cols,options.img_ch)

	model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape = input_shape ))
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

	model.add(Dense(options.nb_classes))
	model.add(Activation('sigmoid'))

	return model

def vggNet():
	model = Sequential()

	if K.image_dim_ordering() == 'th':
		input_shape = (options.img_ch,options.img_rows,options.img_cols)
	else:
		input_shape = (options.img_rows,options.img_cols,options.img_ch)

	model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape = input_shape))
	model.add(Activation('relu'))

	model.add(Convolution2D(32, (3, 3)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(64,(3, 3), padding='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, (3, 3),padding='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(128,(3, 3), padding='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, (3, 3),padding='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, (3, 3),padding='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(256,(3, 3), padding='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, (3, 3),padding='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, (3, 3),padding='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(options.nb_classes))
	model.add(Activation('softmax'))

	return model
