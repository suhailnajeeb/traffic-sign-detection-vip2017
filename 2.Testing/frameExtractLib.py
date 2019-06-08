import cv2
import glob
import threading
import numpy as np
from effectClass.effectClassLib import *
import effectClass.dehaze as dehaze
from options import optionsRCNN


from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage import data,img_as_ubyte,img_as_float,color,restoration,exposure
from skimage.util import random_noise
from skimage._shared._warnings import expected_warnings

from scipy.signal import convolve2d as conv2
from pylab import *
import os

actions = {}												# dictionary will contain preprocessing functions
action  = lambda f:actions.setdefault(f.__name__,f)			# annoymous functions to store preprocessing functions in dictionary

# making generators thread safe with this decorator class
class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()

# decorator for generator functions
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

# get frames from video
@threadsafe_generator
def frameGenerator1(videos,vid_fldr,classify=False):
	try:
		if classify:
			settings = optionsRCNN()		# setting up challenge type classifier
			model = setupRCNN(settings)

		for vidname in videos:

			if classify:
				count = challengeCount()		# iniitating  class to keep track of challenges and required actions

			video_path = vid_fldr + vidname + '.mp4'
			vid = cv2.VideoCapture(video_path)

			success,frame = vid.read()

			fidx = 0		# frame index
			while success:
				fidx +=1

				if classify:
					y = classifyChallenge(model,settings,frame)		# classifying challenge type
					count.incCount(y)
					## print count.challengeType

					if fidx>6:
						frame = actions[count.takeAction](frame)	#  taking appropriate corrective action (unused, just returns the input frame)

					yield count.challengeType,frame

				if not classify:
					yield frame

				if fidx == 300:
					break

				success,frame = vid.read()

	except Exception as e:
		print "\nError: %s\n" %e

# get frames from folder with extracted frames
@threadsafe_generator
def frameGenerator2(vidname,frame_fldr,classify=False):
	try:
		if classify:
			settings = optionsRCNN()				# setting up challenge type classifier
			model = setupRCNN(settings)

		for vid in vidname:

			if classify:
				count = challengeCount()			# iniitating  class to keep track of challenges and required actions

			img_path = frame_fldr + vid
			framepaths = sorted(glob.glob(img_path + '/*.jpg'))		# getting frame paths

			fidx = 0			# frame index

			for path in framepaths:
				fidx +=1
				frame = cv2.imread(path)

				if classify:
					y = classifyChallenge(model,settings,frame)		# classifying challenge type
					count.incCount(y)

					if fidx>6:
						frame = actions[count.takeAction](frame)	# taking appropriate corrective action (unused, just returns the input frame)

					yield count.challengeType,frame

				if not classify:
					yield frame

				if fidx == 300:
					break

	except Exception as e:
		print "\nError: %s\n" %e


# ----------- Preprocessing Functions (Unused) -------------------------------

@action
def adjustExp(img):
	return img
	## print "Adjusting Exposure"

	p2, p98 = np.percentile(img, (5,95))
	processed_image = exposure.rescale_intensity(img, in_range=(p2, p98))

	return processed_image

@action
def adjustGamma(img,gamma=2.1):
	return img
	## print "Adjusting Gamma"

	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)


@action
def deBlur(img):			# unused
	return img

	## print "Deblurring"

	image_gray = img
	psf = np.ones((3,3)) /9
	image_gray = conv2(image_gray, psf, 'same')

	# Restore Image using Richardson-Lucy algorithm
	deconvolved_RL = restoration.richardson_lucy(image_gray, psf, iterations=10)

	return img

@action
def deNoise(img):			# unused
	return img

	## print "Denoising"

	img = cv2.medianBlur(img,11)
	img = cv2.fastNlMeansDenoisingColored(img,None,10,10,block_size,window_size)

	return img

@action
def deHaze(img):			# unused
	return img

	## print "Dehazing"

	I = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	I = img_as_float(I)

	w = 4 					# window size (positive integer) for determing the dark channel and the transition map
	w2 = 3*w 				# window size (positive integer) for guided filter
	strength = 0.90			# strength of the dahazing effect 0 <= stength <= 1 (is 0.95 in the original paper)

	dark = dehaze.dark_channel(I, w)

	haze_pixel = dark >= percentile(dark, 98)
	In = sum(I, axis=2)/3

	bright_pixel = In >= percentile(In[haze_pixel], 98)
	A0 = mean(I[logical_and(haze_pixel, bright_pixel), :], axis=0)

	t = dehaze.transition_map(I, A0, w, strength)

	t = dehaze.box_min(t, w)
	t = dehaze.guidedfilter(I, t, w2, 0.001)
	t[t<0.025] = 0.025

	J=I/t[:, :, np.newaxis] - A0[np.newaxis, np.newaxis, :]/t[:, :, np.newaxis] + A0

	J[J<0]=0
	J[J>1]=1

	with expected_warnings(['precision']):
		J = img_as_ubyte(J)

	J = cv2.cvtColor(J,cv2.COLOR_RGB2BGR)
	return J

@action
def deSnow(img):		# unused
	return img

	## print "Desnowing"

	proc_image1 = actions['deNoise'](img)
	proc_image2 = actions['deBlur'](proc_image1)

	return proc_image2

@action
def doNothing(img):
	return img

	## print "Doing nothing"
