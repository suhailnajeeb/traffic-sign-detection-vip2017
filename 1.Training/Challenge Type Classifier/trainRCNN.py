
from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras_rcnn import *
import cv2
import os
import numpy as np

#-------------------------------- Settings -------------------------------------
# folder where features from extractFeatures.py are saved
feat_dir = './Features/'
model_dir = './Models/'

# load/train old model
load_model = 0
continue_training = 0
evaluate_on_test_data = 0

# model description
model_name = 'BC'
load_model_name = ''

# name of features to load
train_feat_name = ['B_train_125_125','C_train_125_125']		# used for training
test_feat_name = ['B_test_125_125','C_test_125_125']		# used for validation

# Training Parameters
batch_size = 50
epochs = 50
patience = 15			# early stopping parameter
min_delta = 0.01		# minimum change requqired to prevent stopping

# RCNN Parameters
nbRCL = 6				# 6 recurrent layers
nbFilters= 128			# 128 filters in conv layer
filtersize = 3			# 3x3 kernel size

# image parameters
nbChannels = 3
shape1 = 125			# model expects 125x125 images
shape2 = 125
nbClasses = 11			# 11 challenge type classes (see extractFeatures.py)

# -----------------------------------------------------------------------------
# creates directory
def ensur_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

# loads features
def load_feat(feat_dir,train_feat_name):
	trainData = np.load(feat_dir + train_feat_name + '.npz')
	x_train = trainData['X_train']
	y_trainb = trainData['Y_train']
	y_train = np_utils.to_categorical(y_trainb.astype(int),nbClasses)
	trainData.close()
	return x_train,y_train

model_fldr = model_dir + model_name + "/"
model_weights = model_fldr + "weights/"
weights_dir = model_dir + load_model_name + "/best.hdf5"

ensur_dir(model_weights)

if continue_training or not load_model:
	print "\nLoading Training Data"
	print "----------------------\n"

	print "Loading: %s" % train_feat_name[0]
	x_train,y_train = load_feat(feat_dir,train_feat_name[0])


	for feat in train_feat_name[1:]:
		print "Loading: %s" % feat
		x,y = load_feat(feat_dir,feat)
		x_train = np.concatenate((x_train, x), axis=0)
		y_train = np.concatenate((y_train, y), axis=0)

if evaluate_on_test_data:
	print "\nLoading Test Data"
	print "----------------------\n"

	print "Loading: %s" % test_feat_name[0]
	x_test,y_test = load_feat(feat_dir,test_feat_name[0])


	for feat in test_feat_name[1:]:
		print "Loading: %s" % feat
		x,y = load_feat(feat_dir,feat)
		x_test = np.concatenate((x_test, x), axis=0)
		y_test = np.concatenate((y_test, y), axis=0)

	print "\n----------------------\n"

# creating model
model = makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=nbRCL, nbFilters=nbFilters, filtersize = filtersize)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])

# loading weights only to test
if load_model == 1 and continue_training == 0:
	print"\n\nLoading Model:\t %s" % load_model_name
	model.load_weights(weights_dir)

# loading weights and continue training
elif load_model ==1 and continue_training ==1:
	print"\n\nLoading Model:\t %s" % load_model_name
	model.load_weights(weights_dir)

	print"\nContinuing Trianing : \n\n"
	check1 = ModelCheckpoint(model_weights + model_name + "_weights.{epoch:02d}-{val_categorical_accuracy:.3f}.hdf5", monitor='val_categorical_accuracy', save_best_only=True,save_weights_only=True, mode='auto')
	check2 = ModelCheckpoint(model_fldr + "best.hdf5", monitor='val_categorical_accuracy', save_best_only=True,save_weights_only=True, mode='auto')
	check3 = EarlyStopping(monitor='val_categorical_accuracy', min_delta=min_delta, patience=patience, verbose=0, mode='auto')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1,callbacks=[check1,check2,check3],shuffle=True,validation_split=0.25)

# training new model
else:
	print"\nTraining New Model:\t %s\n\n" % model_name
	check1 = ModelCheckpoint(model_weights + model_name + "_weights.{epoch:02d}-{val_categorical_accuracy:.3f}.hdf5", monitor='val_categorical_accuracy', save_best_only=True,save_weights_only=True, mode='auto')
	check2 = ModelCheckpoint(model_fldr + "best.hdf5", monitor='val_categorical_accuracy', save_best_only=True,save_weights_only=True, mode='auto')
	check3 = EarlyStopping(monitor='val_categorical_accuracy', min_delta=min_delta, patience=patience, verbose=0, mode='auto')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1,callbacks=[check1,check2,check3],shuffle=True,validation_split=0.25)

# Evaluating on test data
if evaluate_on_test_data == 1:
	loss,scores = model.evaluate(x_test,y_test,verbose=1)
	print "\n\n\n\tAccuracy on Test Data: %2.2f %%\n\n" %(scores*100)
