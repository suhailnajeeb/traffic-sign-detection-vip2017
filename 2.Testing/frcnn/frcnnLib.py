
import numpy as np
import pickle

import frcnn.config as config
import frcnn.roi_helpers as roi_helpers
import frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model


# creates FRCNN model
def setupFRCNN(options):

	# suppressing tensorflow warnings and allowing gpu memory allocation to grow
	import os
	import tensorflow as tf
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	configTF = tf.ConfigProto()
	configTF.gpu_options.allow_growth = True
	sess = tf.Session(config=configTF)

	# setting system recursion limit
	import sys
	sys.setrecursionlimit(40000)

	# loading config file
	with open(options.config_filename, 'r') as f_in:
		C = pickle.load(f_in)

	# turn off any data augmentation at run time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False

	# obtaining classes used to train classifier
	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	class_mapping = {v: k for k, v in class_mapping.iteritems()}
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

	# number of ROIs fed to classifier per batch
	C.num_rois = int(options.num_rois)

	# acccounting for difference in img dimesion order in theano/tensorflow
	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (1024, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, 1024)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# setting up the layers in RPN and CNN classifier
	shared_layers = nn.nn_base(img_input, trainable=True)
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)
	classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

	model_rpn = Model(img_input, rpn_layers)
	model_classifier_only = Model([feature_map_input, roi_input], classifier)
	model_classifier = Model([feature_map_input, roi_input], classifier)

	# loading weights
	model_rpn.load_weights(options.model_weight_path, by_name=True)
	model_classifier.load_weights(options.model_weight_path, by_name=True)

	# compiling models
	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	return C,model_rpn,model_classifier,model_classifier_only

# detects boxes
def findBBox(Q_frcnn,side,C,options,model_rpn,model_classifier_only,overlap=70):

	# correction applied to bbox co-ordinates oif scanning right side of img
	if side == 'R':
		org_shift = 814-overlap/2
	elif side == 'L':
		org_shift = 0

	# obtaining classes used to train classifier
	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	class_mapping = {v: k for k, v in class_mapping.iteritems()}
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


	while not Q_frcnn.empty():
		try:
			# get image from queue
			img = Q_frcnn.get()

			# removing channel means
			X = img[:, :, (2, 1, 0)]
			X = X.astype(np.float32)

			X[:, :, 0] -= C.img_channel_mean[0]
			X[:, :, 1] -= C.img_channel_mean[1]
			X[:, :, 2] -= C.img_channel_mean[2]

			X = np.transpose(X, (2, 0, 1))
			X = np.expand_dims(X, axis=0)


			if K.image_dim_ordering() == 'tf':
				X = np.transpose(X, (0, 2, 3, 1))

			# get the feature maps and output from the RPN
			[Y1, Y2, F] = model_rpn.predict(X)

			# convert rpn output into co-ordinates of corners of bbox
			R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=options.non_maxima_suprresion_threshold-0.2)

			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			# apply the spatial pyramid pooling to the proposed regions
			bboxes = {}
			probs = {}

			for jk in range(R.shape[0]//C.num_rois + 1):
				ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
				if ROIs.shape[1] == 0:
					break

				if jk == R.shape[0]//C.num_rois:
					# padding R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded

				# passing proposed ROIs to classifier
				[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

				for ii in range(P_cls.shape[1]):

					if np.max(P_cls[0, ii, :]) < options.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
						continue

					cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

					if cls_name not in bboxes:
						bboxes[cls_name] = []
						probs[cls_name] = []

					(x, y, w, h) = ROIs[0, ii, :]

					cls_num = np.argmax(P_cls[0, ii, :])

					try:
						(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
						tx /= C.classifier_regr_std[0]
						ty /= C.classifier_regr_std[1]
						tw /= C.classifier_regr_std[2]
						th /= C.classifier_regr_std[3]
						x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
					except:
						pass

					rpns = C.rpn_stride

					bboxes[cls_name].append([rpns*x, rpns*y, rpns*(x+w), rpns*(y+h)])
					probs[cls_name].append(np.max(P_cls[0, ii, :]))

			bboxForFrames = []

			for key in bboxes:
				bbox = np.array(bboxes[key])
				new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=options.non_maxima_suprresion_threshold-0.2)

				for jk in range(new_boxes.shape[0]):
					(x1, y1, x2, y2) = new_boxes[jk,:]

					x1 += org_shift				# adjusting for change in origin when dividing image into two halves
					x2 += org_shift				# co-ordinates system of righ half shifted
					y1 = 0 if y1<0 else y1		# negative co-ordinates clamped to zero
					y2 = 0 if y2<0 else y2
					bboxForFrames.append((x1,y1,x2,y2))

			Q_frcnn.task_done()
			yield bboxForFrames

		except StopIteration:
			return
