
from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras_rcnn import *
import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
configTF = tf.ConfigProto()
configTF.gpu_options.allow_growth = True
sess = tf.Session(config=configTF)

class challengeCount:
	def __init__(self):
		self.effects_combined = {0:0,1:0,3:0,5:8,2:1,4:2,6:3,7:4,8:5,9:6,10:7}																	# some challenges are  grouped
		self.effects_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}																				# count for each grouped class
		self.effects_class = {0:'NoCh',1:'Blurry',2:'Dark',3:'Bright',4:'Noisy',5:'Shadow',6:'Snow',7:'Haze',8:'DirtyLens'}						# class categories
		self.action = {0:'doNothing',1:'deBlur',2:'adjustGamma',3:'adjustExp',4:'deNoise',5:'doNothing',6:'deSnow',7:'deHaze',8:'doNothing'}
		self.challengeType = 'doNothing'  #
		self.takeAction = 'doNothing'	#

	def incCount(self,y):
		self.effects_count[self.effects_combined[y]] = self.effects_count[self.effects_combined[y]] + 1		# increasing count
		self.challengeType = self.effects_class[max(self.effects_count,key=self.effects_count.get)]			# updating with challengeType having highest count
		self.takeAction = self.action[max(self.effects_count,key=self.effects_count.get)]					# updating appropriate action

# setting up challenge classifier
def setupRCNN(options):
	model = makeModel(options.nbChannels, options.shape1, options.shape2, options.nbClasses, nbRCL=options.nbRCL, nbFilters=options.nbFilters, filtersize =options.filtersize)
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
	model._make_predict_function()
	model.load_weights(options.weights_dir)
	return model

# classifying challenge
def classifyChallenge(model,options,img):
	x = img[418:818,614:1014]			# taking sample section from  middle of frame
	x = cv2.resize(x,(options.shape1, options.shape2),interpolation =cv2.INTER_CUBIC)
	x = np.expand_dims(x, axis=0)
	y = model.predict(x)
	y = np.argmax(y)
	return y
