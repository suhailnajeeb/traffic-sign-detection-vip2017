import cv2
import numpy as np
import cPickle as pickle
from keras.utils import generic_utils

def get_data(input_path,config,name,num_imgs):
	found_bg = False
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	visualise = True

	with open(input_path,'r') as f:

		print('\nParsing annotation files\n')
		progbar = generic_utils.Progbar(num_imgs)
		iter_num = 0

		for line in f:
			line_split = line.strip().split(',')
			(filename,x1,y1,x2,y2,class_name) = line_split

			# checking if class already accounted for
			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:

				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True

				# assigning value to each class
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:

				# creating empty entry in dict corresponding to filename
				all_imgs[filename] = {}

				# reading and cropping image
				img = cv2.imread(filename)
				img = img[config.cutImage[0]:config.cutImage[1],config.cutImage[2]:config.cutImage[3]]

				# saving image info to dict
				(rows,cols) = img.shape[:2]
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []

				# randomly placing image in train / validation set
				## if np.random.randint(0,6) > 0:
				# 	all_imgs[filename]['imageset'] = 'trainval'
				## else:
				## 	all_imgs[filename]['imageset'] = 'test'

				all_imgs[filename]['imageset'] = 'trainval'


			# saving bounding box (ground truth) info and class
			# if multiple ROI, that info is also appeneded to the bboxes column for that image
			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

			iter_num +=1
			progbar.update(iter_num)

		# appending all the entries in all_imgs into all data
		all_data = []

		for key in all_imgs:
			all_data.append(all_imgs[key])

		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch

		print '\nDumping Trainset Data into Pickle File ... \n'
		pickle.dump(all_data,open('./' + name + '/' + name + '_all_data.p','wb'))
		pickle.dump(classes_count,open('./' + name + '/' + name + '_classes_count.p','wb'))
		pickle.dump(class_mapping,open('./' + name + '/' + name + '_classes_mapping.p','wb'))

		return all_data, classes_count, class_mapping

def load_data(name):
	print '\nLoading Trainset Data from Pickle File ... \n'
	all_data = pickle.load(open('./' + name + '/' + name + '_all_data.p','rb'))
	classes_count = pickle.load(open('./' + name + '/' + name + '_classes_count.p','rb'))
	class_mapping = pickle.load(open('./' + name + '/' + name + '_classes_mapping.p','rb'))
	return all_data, classes_count, class_mapping
