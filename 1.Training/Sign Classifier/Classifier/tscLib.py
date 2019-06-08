
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import numpy as np
import glob
import os
import logging
import matplotlib.pyplot as pl
import tscModels
from tscParams import *
import random

scores = 0
SC = None
# ------------------------------------ Functions -------------------------------

# function to create directory
def ensur_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

# creates sequential model SC
def createModel(arch):
	return getattr(tscModels,arch)()

def loadWeights(model,weights,model_name=model_name):
	print "\nLoading Weights for Model %s\t" % model_name,
	model.load_weights(weights)
	print "\tDone\n"

def trainBatch(model,datagen,steps,valgen,stepsval):
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])


	check = ModelCheckpoint(model_weights + model_name + "_weights.{epoch:02d}-{categorical_accuracy:.3f}.hdf5", monitor='val_categorical_accuracy', save_best_only=True,save_weights_only=True, mode='auto')
	check2 = ModelCheckpoint(model_fldr + "best.hdf5", monitor='val_categorical_accuracy', save_best_only=True,save_weights_only=True, mode='auto')
	check3 = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.03, patience=3, verbose=0, mode='auto')


	SCModel = model.fit_generator(datagen, steps, epochs=nb_epoch, callbacks=[check,check2,check3], validation_data = valgen, validation_steps = stepsval )
	return SCModel

def countEpoch(name_train):
	epochCount = 0
	print "Evaluating Epochs!"
	feat_loc = []
	for name in name_train:
		feat_loc.append((feat_dir + name + '.npz'))
	for feat in feat_loc :
		npzfile = np.load(feat)
		y_train = npzfile['Y_train']
		n = len(y_train)/batch_size
		epochCount += n
		#print "Data: %d" %(datacount)
		npzfile.close()
	print "Total Steps Per Epoch: %d" %(epochCount)
	return epochCount


def to_int(y):
	signs = []
	R,C = y.shape
	for i in range(0,R):
		for j in range(0,C):
			if y[i,j] == 1:
				signs.append(j+1)
				break
	return signs

def data_generator():
	while True:
		feat_loc = []
		for name in name_train:
			feat_loc.append((feat_dir + name + '.npz'))
		#j = np.random.permutation(len(feat_loc))
		#feat_loc = feat_loc[j]
		#random.shuffle(feat_loc)
		for feat in feat_loc : 
			print "\nloading :%s" %(feat)
			npzfile = np.load(feat)
			x_train = npzfile['X_train']
			y_trainb = npzfile['Y_train']
			npzfile.close()
			y_train = np_utils.to_categorical(y_trainb.astype(int)-1, nb_classes)
			x_train = x_train.astype('float32')
			x_train /= 255
			i = np.random.permutation(len(x_train))
			x_train = x_train[i]
			y_train = y_train[i]		
			n=len(x_train)-batch_size
			start = 0
			while (start<n):
				x_batch = x_train[start:start+batch_size]
				y_batch = y_train[start:start+batch_size]
				start += batch_size		
				yield (x_batch,y_batch)


def logModel(model_history=None):
	if continue_training is 1:
		logging.basicConfig(filename = model_fldr + model_name + '.log',level=logging.INFO)
		logging.info("\n--------------------- Continuing Training --------------------\n")
	else:
		logging.basicConfig(filename = model_fldr + model_name + '.log',filemode = 'w',level=logging.INFO)
	logging.info ("\tName: %s" % model_name)
	logging.info ("\tDescription: %s" % desc)
	logging.info ("\tImage Size:\t %d x %d" % (img_rows,img_cols))
	logging.info ("\tBatch Size:\t %d" % batch_size)
	logging.info ("\tEpochs:\t\t %d" % nb_epoch)
	logging.info ("\tTrain Set:\t %s" % name_train )

	if test_set:
		logging.info ("\tTest Set:\t %s" % name_test )

	if load_weights == 0:
		logTraining(model_history)
	elif load_weights == 1:
		logTesting()


def logTraining(FCmodel):
	logging.basicConfig(filename = model_fldr + model_name + '.log',level=logging.INFO)
	logging.info("\n\nAccuracy on Training data after each epoch:\n")
	logging.info(FCmodel.history["categorical_accuracy"])
	pl.figure()
	pl.plot(FCmodel.history[ 'loss' ])
	pl.title( 'model loss' )
	pl.ylabel( 'loss' )
	pl.xlabel( 'epoch' )
	pl.savefig( model_dir + model_name + '/' +model_name +  '_loss.png')
	pl.figure()
	pl.plot(FCmodel.history[ 'categorical_accuracy' ])
	pl.title( 'model accuracy' )
	pl.ylabel( 'accuracy' )
	pl.xlabel( 'epoch' )
	pl.legend([ 'train' , 'test' ], loc= 'upper left' )
	pl.savefig( model_dir + model_name + '/' +model_name + '_acc.png')
	pl.show()

def logTesting():
	logging.basicConfig(filename = model_fldr + model_name + '.log',level=logging.INFO)
	logging.info("\n\nAccuracy on Test Data:\n\t: %.2f%%" %(scores*100))

def val_generator():
	while True:
		feat_loc = []
		for name in name_test:
			feat_loc.append((feat_dir + name + '.npz'))
		random.shuffle(feat_loc)
		for feat in feat_loc : 
			print "\nloading :%s" %(feat)
			npzfile = np.load(feat)
			x_train = npzfile['X_train']
			y_trainb = npzfile['Y_train']
			npzfile.close()
			y_train = np_utils.to_categorical(y_trainb.astype(int)-1, nb_classes)
			x_train = x_train.astype('float32')
			x_train /= 255
			i = np.random.permutation(len(x_train))
			x_train = x_train[i]
			y_train = y_train[i]		
			n=len(x_train)-batch_size
			start = 0
			while (start<n):
				x_batch = x_train[start:start+batch_size]
				y_batch = y_train[start:start+batch_size]
				start += batch_size		
				yield (x_batch,y_batch)
