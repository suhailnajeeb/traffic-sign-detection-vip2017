
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

# build and return model
def makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=5,nbFilters=128, filtersize = 3):

	model = BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)
	return model


# model architecture
def BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize):

	def RCL_block(l_settings, l, pool=True, increase_dim=False,layer_num=None):

		## if layer_num==1:
			## print "\nCreating Recurrent blocks ...",

		input_num_filters = l_settings.output_shape[1]

		if increase_dim:
				out_num_filters = input_num_filters*2

		else:
				out_num_filters = input_num_filters

		conv1 = Conv2D(out_num_filters, 3, strides=3, padding='same',data_format='channels_last')
		stack1 = conv1(l)
		stack2 = BatchNormalization()(stack1)
		stack3 = PReLU()(stack2)

		conv2 = Conv2D(out_num_filters, filtersize, strides=1, padding='same', kernel_initializer = 'he_normal',data_format='channels_last')
		stack4 = conv2(stack3)
		stack5 = add([stack1, stack4])
		stack6 = BatchNormalization()(stack5)
		stack7 = PReLU()(stack6)

		conv3 = Conv2D(out_num_filters, filtersize, strides=1, padding='same', weights = conv2.get_weights(),data_format='channels_last')
		stack8 = conv3(stack7)
		stack9 = add([stack1, stack8])
		stack10 = BatchNormalization()(stack9)
		stack11 = PReLU()(stack10)

		conv4 = Conv2D(out_num_filters, filtersize, strides=1, padding='same', weights = conv2.get_weights(),data_format='channels_last')
		stack12 = conv4(stack11)
		stack13 = add([stack1, stack12])
		stack14 = BatchNormalization()(stack13)
		stack15 = PReLU()(stack14)

		# will pool layers if recurrent layer number multiple of 2
		if pool:
			stack16 = MaxPooling2D((2, 2), padding='same')(stack15)
			stack17 = Dropout(0.1)(stack16)
		else:
			stack17 = Dropout(0.1)(stack15)

		return stack17

	#Build Network
	input_img = Input(shape=(shape1,shape2,nbChannels))
	conv_l = Conv2D(nbFilters, filtersize, strides=filtersize, padding='same', activation='relu',data_format='channels_last')
	l = conv_l(input_img)

	# Feed to recurrent layers; pool at the end of every even recurrent layer
	for n in range(nbRCL):
		if n % 2 ==0:
			l = RCL_block(conv_l, l, pool=False,layer_num=n+1)
		else:
			l = RCL_block(conv_l, l, pool=True,layer_num=n+1)

	out = Flatten()(l)
	l_out = Dense(nbClasses, activation = 'softmax')(out)

	model = Model(inputs = input_img, outputs = l_out)
	## print "Complete."
	return model
