
def trainModel(options):

	import random
	import pprint
	import sys
	import time
	import numpy as np
	import pickle
	import os

	from keras import backend as K
	from keras.optimizers import Adam,SGD,RMSprop
	from keras.layers import Input
	from keras.models import Model
	from frcnn import config, data_generators
	from frcnn import losses as losses
	from frcnn import resnet as nn
	import frcnn.roi_helpers as roi_helpers
	from keras.utils import generic_utils
	from frcnn.simple_parser import get_data
	from frcnn.simple_parser import load_data

	# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	import tensorflow as tf
	configTF = tf.ConfigProto()
	configTF.gpu_options.allow_growth = True
	sess = tf.Session(config=configTF)

	sys.setrecursionlimit(40000)

	# Config class stores all relevant settings to be recalled during test/further training
	C = config.Config()
	C.im_size = options.im_size
	C.anchor_box_scales = options.anchor_box_scales
	C.anchor_box_ratios = options.anchor_box_ratios
	C.num_rois = int(options.num_rois)
	C.use_horizontal_flips = bool(options.horizontal_flips)
	C.use_vertical_flips = bool(options.vertical_flips)
	C.rot_90 = bool(options.rot_90)
	C.rpn_max_overlap = options.rpn_max_overlap_threshold
	C.model_path = options.output_weight_path
	C.balanced_classes = options.balanced_classes
	C.rpn_stride = options.rpn_stride
	C.cutImage = options.cutImage

	# loading old weights to continue training
	if options.load_weights:
		C.base_net_weights = options.input_weight_path

	# get_data() function returns image list along with info on bbox,
	# height, width in all_imgs,and class data in classes_count and class_mapping
	if options.load_data:
		all_imgs, classes_count, class_mapping = load_data(options.name)
	else:
		all_imgs, classes_count, class_mapping = get_data(options.train_path,C,options.name,options.num_frames)

	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)

	C.class_mapping = class_mapping

	# making assigned value of class the key to index the dictionary
	inv_map = {v: k for k, v in class_mapping.iteritems()}

	print('Training images per class:')
	pprint.pprint(classes_count)
	print('Num classes (including bg) = {}'.format(len(classes_count)))
	print '\nUsing RPN Stride = %d' % C.rpn_stride
	config_output_filename = options.config_filename

	with open(config_output_filename, 'w') as config_f:
		pickle.dump(C,config_f)
		print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

	random.shuffle(all_imgs)
	num_imgs = len(all_imgs)

	train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']

	print('\nTraining on {} Frames'.format(len(train_imgs)))

	# ground truth boxes are obtained
	data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
	else:
		input_shape_img = (None, None, 3)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn = nn.rpn(shared_layers, num_anchors)

	classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

	model_rpn = Model(img_input, rpn[:2])
	model_classifier = Model([img_input, roi_input], classifier)

	# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
	model_all = Model([img_input, roi_input], rpn[:2] + classifier)

	try:
		model_rpn.load_weights(C.base_net_weights, by_name=True)
		model_classifier.load_weights(C.base_net_weights, by_name=True)
		print('\nLoaded weights from {}'.format(C.base_net_weights))
	except:
		print('\nNo pretrained weights found in folder...')
		print('Proceeding to train from scratch\n\n')

	# setting learning rates and optimizers
	optimizer = Adam(lr=1e-4)
	optimizer_classifier = Adam(lr=1e-4)

	model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
	model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

	model_all.compile(optimizer='sgd', loss='mae')

	# if epoch_length was set as default, a epoch goes over entire trainset images
	if options.epoch_length == 'default':
		epoch_length = len(train_imgs)

	# else it uses the number of images given
	else:
		epoch_length = int(options.epoch_length)

	num_epochs = int(options.num_epochs)
	iter_num = 0

	losses = np.zeros((epoch_length, 5))
	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []
	start_time = time.time()

	best_loss = np.Inf

	class_mapping_inv = {v: k for k, v in class_mapping.iteritems()}
	print('Starting training\n')

	prev_pos_samples = []
	prev_neg_samples = []

	for epoch_num in range(num_epochs):

		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

		while True:
			try:
				if iter_num == epoch_length and C.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []

					if mean_overlapping_bboxes == 0:
						print('\n\nRPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.\n')

				# date_gen_train is a generator defined in data_generators.py
				# it reads the the image with the filepath in image list and returns relevant data for training
				X, Y, img_data = data_gen_train.next()

				# train_on_batch() is the keras function defined in Sequential.
				# does a single gradient update on given data (essentially, one epoch)
				loss_rpn = model_rpn.train_on_batch(X, Y)

				# uses trained model to make prediction
				P_rpn = model_rpn.predict_on_batch(X)

				# converting RPN prediction to ROI
				R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

				# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
				X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

				# if no ROI is detected
				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue

				neg_samples = np.where(Y1[0, :, -1] == 1)
				pos_samples = np.where(Y1[0, :, -1] == 0)

				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []

				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []

				rpn_accuracy_rpn_monitor.append(len(pos_samples))
				rpn_accuracy_for_epoch.append((len(pos_samples)))

				if C.num_rois > 1:
					# Take half of positive samples and half of negative samples for classfier training

					if len(pos_samples) < C.num_rois/2:
						selected_pos_samples = pos_samples.tolist()

					else:
						selected_pos_samples = np.random.choice(pos_samples, C.num_rois/2, replace=False).tolist()

					try:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()

					except:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

					sel_samples = selected_pos_samples + selected_neg_samples

				else:
					# in the extreme case where num_rois = 1, we pick a random pos or neg sample
					selected_pos_samples = pos_samples.tolist()
					selected_neg_samples = neg_samples.tolist()

					if np.random.randint(0, 2):
						sel_samples = random.choice(neg_samples)
					else:
						sel_samples = random.choice(pos_samples)

				loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

				losses[iter_num, 2] = loss_class[1]
				losses[iter_num, 3] = loss_class[2]
				losses[iter_num, 4] = loss_class[3]
				losses[iter_num, 0] = loss_rpn[1]
				losses[iter_num, 1] = loss_rpn[2]

				iter_num += 1

				progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
										  ('detector_cls', np.mean(losses[:iter_num, 2])), ('rpn_overlap', float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor))])

				if iter_num == epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
					rpn_accuracy_for_epoch = []

					if C.verbose:
						print('\n---------------------------------------------------------------------------------------')
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))
						print('Elapsed time: {}'.format(time.time() - start_time))


					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					iter_num = 0
					start_time = time.time()

					if curr_loss < best_loss:

						if C.verbose:
							print('\nTotal loss decreased from {} to {}'.format(best_loss,curr_loss))
							print('---------------------------------------------------------------------------------------\n')
						best_loss = curr_loss

						# saving weights with smallest loss (overwrite)
						model_all.save_weights(C.model_path)
					else :
						if C.verbose:
							print('\nLoss did not improve')
							print('---------------------------------------------------------------------------------------\n')


					# also saving weights for each epoch
					model_all.save_weights(C.model_path[:-5] + '_%03d'%(epoch_num+1) + '_%2.2f'%mean_overlapping_bboxes + '.hdf5')

					break

			except Exception as e:
				print('\nException: {}\n'.format(e))
				continue


	print('\n-----------------\nTraining complete!\n')
