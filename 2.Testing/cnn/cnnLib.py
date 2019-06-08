
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import numpy as np
import cv2
import glob
import cnnModels

# ------------------------------------ Functions -------------------------------


# creates sequential model SC
def createModel(options):
	return getattr(cnnModels,options.model_arch)(options)


# loads saved weights into a created model
def loadWeights(model,weights_path):
	model.load_weights(weights_path)


# scaling down bounding boxes
def scaleBoxes(points,factor):

	if factor ==1:
		return points

	factor = 1 - factor

	x1,y1 = points[0],points[1]
	x2,y2 = points[2],points[3]

	xn1 = int(x1 + (factor*(x2-x1))/2)
	xn2 = int(x2 - (factor*(x2-x1))/2)
	yn1 = int(y1 + (factor*(y2-y1))/2)
	yn2 = int(y2 - (factor*(y2-y1))/2)

	return [xn1,yn1,xn2,yn2]

# predict class of image input
def classifySign(model,cnnQ,cnnOutQ,imgQ,options,class_map=None,scale=1,showROI=False):

	while not imgQ.empty():
		try:
			img = imgQ.get()		# get frame
			ROIs = cnnQ.get()		# get bounding boxes

			if ROIs == []:				# if no bounding boxes, return [0]
				imgQ.task_done()
				cnnQ.task_done()
				yield [0]
				continue

			x=[]
			order_adj = []
			for i,roi2 in enumerate(ROIs):
				roi = scaleBoxes(roi2,scale)
				x1 = 0 if roi[0]<0 else 1627 if roi[0]>1627 else roi[0]
				y1 = 0 if roi[1]<0 else 1235 if roi[1]>1235 else roi[1]
				x2 = 0 if roi[2]<0 else 1627 if roi[2]>1627 else roi[2]
				y2 = 0 if roi[3]<0 else 1235 if roi[3]>1235 else roi[3]

				if x2-x1 == 0 or y2-y1 == 0:									# if box size = skip but keep index in order_adj to be used later to adjust outputs
					order_adj.append(i)
					continue

				imgcrop = img[y1:y2,x1:x2]
				imgresize = cv2.resize(imgcrop,(options.img_rows,options.img_cols),interpolation=cv2.INTER_CUBIC)
				x.append(imgresize)

				if showROI:
					cv2.imshow('roi%d'%i,imgresize)
					cv2.waitKey(1)

			if x == []:						# if no valid images (bbox size = 0), yield 0 and continue
				imgQ.task_done()
				cnnQ.task_done()
				cnnOutQ.put([0])
				yield [0]
				continue

			x = np.array(x)
			x = x.astype('float32')
			x /= 255
			y = model.predict_on_batch(x)

			Y = []
			for yout in y:
				Y.append(np.argmax(yout)+1)		# converting categorical array to row

			j = 0
			Y_adj = []
			for i in range(len(ROIs)):
				if i in order_adj:
					Y_adj.append(0)				# filling in 0s for boxes that were not valid
				else:
					Y_adj.append(Y[j])
					j+=1

			if class_map is None:
				imgQ.task_done()
				cnnQ.task_done()
				cnnOutQ.put(Y_adj)
				yield Y_adj

			else:
				y2 = [class_map[num] for num in Y_adj]
				imgQ.task_done()
				cnnQ.task_done()
				cnnOutQ.put(y2)
				yield y2

		except StopIteration:
			return
