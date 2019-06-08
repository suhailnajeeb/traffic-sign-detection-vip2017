

# Settings for FRCNN + RPN (bounding box detector)
class optionsFRCNN:
	def __init__(self):

		# model_name
		self.name = 'All_7class_[50,150]_660_R'

		# batch size for classifier
		self.num_rois = 50

		# amount of overlap in order to suprress non maxima
		self.non_maxima_suprresion_threshold = 0.7

		# probability thresold for supressing windows through classifier
		self.bbox_threshold = 0.84

		# location of model weights and config file
		self.model_weight_path = './frcnn/' + self.name + '/' + self.name + '_model_frcnn.hdf5'
		self.config_filename = './frcnn/' + self.name + '/' +self.name + '_config.pickle'


# Settings for CNN Classifier	(sign Classifier)
class optionsCNN:
	def __init__(self):

		# model_name
		self.model_weights_default = './cnn/default/best.hdf5'

		# model weights for different challenge types
		self.model_weights_effects = {	'NoCh'		:	'./cnn/NoCh/best.hdf5',
									  	'Blurry'	:	'./cnn/Blur/best.hdf5',
									  	'Dark' 		:	'./cnn/Darkening/best.hdf5',
									  	'Bright' 	:	'./cnn/Exposure/best.hdf5',
									  	'Noisy' 	:	'./cnn/Noise/best.hdf5',
									  	'Shadow'	:	'./cnn/Shadow/best.hdf5',
									  	'Snow'		:	'./cnn/Snow/best.hdf5',
									  	'Haze'		:	'./cnn/Haze/best.hdf5',
									  	'DirtyLens'	:	'./cnn/DirtyLens/best.hdf5'		}

		# model architecture to use, defined in cnnModels.py
		self.model_arch = 'sc_v6'

		# input image parameters
		self.img_rows = 64
		self.img_cols = 64
		self.img_ch = 3

		# Traning parameters
		self.nb_epoch = 10
		self.batch_size = 50
		self.patience = 3
		self.nb_classes = 24


# Settings for RCNN Classifier (challenge type classifier)
class optionsRCNN:
	def __init__(self):

		# RCNN Parameters
		self.nbRCL= 6				# recuurent layers
		self.nbFilters= 128			# number of filters in conv layers
		self.filtersize = 3			# filter kernel size

		# image parameters
		self.nbChannels = 3
		self.shape1 = 125			# x
		self.shape2 = 125			# y
		self.nbClasses = 11

		# weights directory
		self.weights_dir = "./effectClass/best.hdf5"

		# Training Parameters
		self.batch_size = 50
		self.epochs = 50
		self.patience = 15
		self.min_delta = 0.01
