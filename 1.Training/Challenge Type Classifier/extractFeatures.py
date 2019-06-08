
import cv2
import numpy as np
import glob
import os

# ---------------------------------- Settings  ---------------------------------

# location of extracted Frames
loc = '../Frames/'

# save directory for features
feat_dir = './Features/'

# Video Selection
# Set N_vid = [0,0] to process only train videos, and N_vid = [0,1] to process only test videos
# if you want to list particular videos, put N_vid[0] = -1 and the follow with video # for rest of the elements
# else if you want a particular range, just enter the range in N_vid; e.g [1,9] will process video # 1 to video # 9

N_vid = [0,1]		# video sequence
effects = [0,12]	# challenge type 1-12 (1-11 for syn automatically adjusted )
levels =[2,3]		# challenge level 1-5
syn = 1				# make 1 to include Synthesized videos
only_syn = 0		# make 1 and syn = 1 to include only Synthesized videos

# name of feature set, and optional description
name = 'B'
desc = ''

# resize parameters
img_rows = 125
img_cols = 125

# challenge types are 12; but they are regrouped
# Lens Blur(2) and Gaussian Blur(7) constitute class 2
# Rain(9) and Noise(8) constitute class 7

class_map = {	0:0,	# No Ch
				1:1,	# Decolored
				2:2,	# Blur (lens)
				3:3,	# Codec Error
				4:4,	# Darkening
				5:5,	# Dirty Lens
				6:6,	# Exposure
				7:2,	# Blur (gaussian)
				8:7,	# Noise
				9:7,	# Noise (rain)
				10:8,	# Shadow
				11:9,	# Snow
				12:10	# Haze
}
# ---------------------------------- Code --------------------------------------

if N_vid[0] is 0 and N_vid[1] is 0:
	typ = 'train'
elif N_vid[0] is 0 and N_vid[1] is 1:
	typ = 'test'
else:
	typ = 'train'

# full save name of feature
train_feat = feat_dir + name + '_' + typ + '_' + str(img_rows) + '_' + str(img_cols)

# Glob accumalator Arrays
X_train=[]        # pixels
Y_train=[]        # labels

# test & train videos - given in the VIP Cup 2017 pdf
train_vid_rea =[1,2,3,9,10,11,12,13,14,15,16,17,20,22,23,25,27,28,29,30,32,33,34,35,36,37,40,42,43,44,45,46,48,49]
train_vid_syn = [1,3,5,7,8,10,11,14,15,19,21,23,24,25,26,27,29,30,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49]
test_vid_rea = [4,5,6,7,8,18,19,21,24,26,31,38,39,41,47]
test_vid_syn = [2,4,6,9,12,13,16,17,18,20,22,28,31,32,36]

idx = np.arange(0,300)

# function to make directories
def ensur_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

# t is type of video (real/synthesized)
for t in range (1+only_syn,2+syn):

	# Real
	if t == 1:

		# deteriming which videos to take based on parameters
		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_rea					# Train
		elif N_vid[0] == 0 and N_vid[1] == 1:
			N = test_vid_rea					# Test
		elif N_vid[0] == -1:
			N = N_vid[1:]						# Custom, Individual
		elif N_vid[0]>0:
			N = range(N_vid[0],N_vid[1]+1)		# Custom, Range

	# Synthesized
	elif t==2 :

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_syn
		elif N_vid[0] == 0 and N_vid[1] == 1:
			N = test_vid_syn
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[1],N_vid[2]+1)

	# iterating over video sequences
	for n in N:
		# iterating over challenges
		for eff in range(effects[0],effects[1]+1):

			if t == 2 and eff == 12:
				continue

			lvl = levels[0]

			# iterating over challenge types
			while lvl < levels[1]+1:

				if eff == 0:
					vidname = "%02d_%02d_00_00_00" % (t,n)
					lvl = 10
				else:
					vidname = "%02d_%02d_01_%02d_%02d" % (t,n,eff,lvl)
					lvl += 1

				print vidname,
				print '\tTook ',

				framepath = sorted(glob.glob(loc + vidname + '/*.jpg'))
				sidx = np.random.choice(idx,replace=False,size = 30)			# sampling frames, taking 30 from each video
				sampleFramePath = [framepath[i] for i in sidx]					# getting path of sampled frames

				print len(sampleFramePath),
				print " Samples"

				for path in sampleFramePath:
					x = cv2.imread(path)
					x = x[418:818,614:1014]														# cropping middle 400x400 segment
					x = cv2.resize(x,(img_rows,img_cols),interpolation =cv2.INTER_CUBIC)		# resizing to 125x125
					X_train.append(x)
					Y_train.append(class_map[eff])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print "\n Saving dataset to disk ..."
np.savez(train_feat, X_train=X_train,Y_train=Y_train)

print "\n Done!"
print "\n %5.1f MB file generated" %(os.path.getsize(train_feat + '.npz')/1000000.0)
