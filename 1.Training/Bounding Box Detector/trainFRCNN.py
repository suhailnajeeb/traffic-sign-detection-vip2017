
import numpy as np
import glob
import os
from train_frcnn import *

# ----------------------------------- Dataset Settings ------------------------------

# Directory containing extracted frames
loc = "../Frames/"

label_dir = './labels2_5/'			# Labels directory (boxes enlarged by 2.5 times in x and y)

# Video Selection
# Set N_vid = [0,0] to process only train videos
# if you want to list particular videos, put N_vid[0] = -1 and the follow with video # for rest of the elements
# else if you want a particular range, just enter the range in N_vid; e.g [1,9] will process video # 1 to video # 9

N_vid = [0,0]
effects = [0,12]	# [0,0] for no challenge
levels =[1,5]		# [1,5] for all levels
syn = 1				# 1 to include Synthesized videos
only_syn = 0		# 1 to extract ONLY Synthesized videos

# for training the  FRCNN, we didn't take all the frames to save on training time
# these parameters dictate the skipping order
frame_skipR = 3							# 1 implies no skipping; else take every nth frame
frame_skipS = 4
skip_level_seq = [[1,2],[2,3],[1,3]]	# these levels will be skipped for consecutive videos in this order


# ----------------------------------- Training Settings ------------------------------
class options:
	def __init__(self):

		# model_name (L implies left, and R implies right)
		self.name = 'All_7class_[50,150]_660_R'
		self.load_name = ''

		# load previous weights to continue Training
		self.load_weights = False

		# load data file already generated previously
		self.load_data = False

		# number of epochs
		self.num_epochs = 40
		# the number of images iterated over per epoch
		self.epoch_length = 12000

		# resize the smallest side of input image to this value
		self.im_size = 660
		# the amount of overlap between the right and left half of the frame
		self.overlap = 70

		# threshold for asserting positive samples
		self.rpn_max_overlap_threshold = 0.7

		# window sizes and ratios used by RPN
		self.anchor_box_scales = [50,150]
		self.anchor_box_ratios = [[1, 1]]
		self.rpn_stride = 16

		if self.name[-1] == 'R':
			self.cutImage = [0,660,814-self.overlap/2,1627]		#(y1,y2,x1,x2)
		elif self.name[-1] == 'L':
			self.cutImage = [0,660,0,814+self.overlap/2]		#(y1,y2,x1,x2)
		else:
			self.cutImage = [0,660,0,1628]		#(y1,y2,x1,x2)

		# batch size for classiifer and ROI pooling
		self.num_rois = 30

		# location of config file to load settings from (default = 'config.pickle')
		self.config_filename = './' + self.name + '/' +self.name + '_config.pickle'

		# location of loading/saving model weights
		self.input_weight_path = './' + self.load_name + '/' + self.load_name + '_model_frcnn.hdf5'
		self.output_weight_path = './' + self.name + '/' + self.name + '_model_frcnn.hdf5'

		# getting background samples
		self.getBGsamples = False

		# options for applying augmentations to train data
		self.horizontal_flips = False
		self.vertical_flips = False
		self.rot_90 = False

		# parameters for trainset list processing
		self.train_path = 'trainset.txt'

		# remapping of sign class in labels
		self.num_frames = None
		self.balanced_classes = True

		#  the 14 different type of signs were classed into thse categories
		self.class_map = {	1:'ABC',				2:'whiteMiddle',
							3:'whiteMiddle',		4:'diagonal',
							5:'diagonal',			6:'stop',
							7:'bike',				8:'triangle',
							9:'diagonal',			10:'diagonal',
							11:'whiteMiddle',		12:'stop',
							13:'triangle',			14:'ABC',
							15:'bg'		}

option = options()

# -------------------------------- Code ---------------------------------------

directory = os.path.dirname('./' + option.name + '/')
if not os.path.exists(directory):
	os.makedirs(directory)

entries = []


# train videos - given in the VIP Cup 2017 pdf
train_vid_rea =[1,2,3,9,10,11,12,13,14,15,16,17,20,22,23,25,27,28,29,30,32,33,34,35,36,37,40,42,43,44,45,46,48,49]
train_vid_syn = [1,3,5,7,8,10,11,14,15,19,21,23,24,25,26,27,29,30,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49]

# keeps track of which level pair has been skipped
skip_state = 0

# t is video type (real / synthesized)
for t in range (1+only_syn,2+syn):

	if t == 1:

		frame_skip = frame_skipR

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_rea
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[0],N_vid[1]+1)

	elif t==2 :

		frame_skip = frame_skipS

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_syn
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[1],N_vid[2]+1)

	for n in N:
		skip_offset = -1		# skipping of frames start from different frames to ensure uniform sampling

		try:
			dataset = np.genfromtxt(label_dir + "%02d_%02d.csv" % (t,n) , delimiter=',')		# loading labels

			if np.isnan(dataset[0,0]):
				l = len(dataset[1:,0])
				llx = dataset[1:,2]
				lly = dataset[1:,3]
				urx = dataset[1:,8]
				ury = dataset[1:,9]
				f_num = dataset[1:,0]
				flags = dataset[1:,1]
			else:
				l = len(dataset[0:,0])
				llx = dataset[0:,2]
				lly = dataset[0:,3]
				urx = dataset[0:,8]
				ury = dataset[0:,9]
				f_num = dataset[0:,0]
				flags = dataset[0:,1]

			skip_level = skip_level_seq[skip_state%len(skip_level_seq)]		# picking level pair to skip
			skip_state+=1
			unq_frames = np.unique(f_num)
		except:
			print "Has no ROI : \t %02d_%02d.csv" %(t,n)	# some video sequences do not contain any ROI
			continue

		for eff in range(effects[0],effects[1]+1):

			if t == 2 and eff == 12:		# synthesized videos only have 11 effects
				continue

			lvl = levels[0]
			skip_offset = (skip_offset + 1)%(frame_skip)

			while lvl < levels[1]+1:

				if eff == 0:
					vidname = "%02d_%02d_00_00_00" % (t,n)

					lvl = 10

					print "Processing: \t",
					print vidname
				else:
					if lvl in skip_level:			# skipping levels as given in a pair inside skip_level_seq
						lvl += 1
						continue

					vidname = "%02d_%02d_01_%02d_%02d" % (t,n,eff,lvl)
					lvl += 1

					print "Processing: \t",
					print vidname

				sample_frames = unq_frames[skip_offset::frame_skip]			# picking frames to sample

				for i in range(l):

					if f_num[i] not in sample_frames:			# if frame not part of list of framees to be sampled, continue
						continue

					# if training with right half of image, the box co-ordinates need to be shifted
					# the origin instead of being at the top left corner is now near the top middle, depending where image is split
					# the place where image is split is dictated by options.cutImage, (see above)
					if option.name[-1] == 'R':

						# if box completely within split right half, include box
						if (llx[i] >= option.cutImage[2]):
							entry = loc + vidname + '/' + "%03d.jpg," % f_num[i]
							entry = entry + str(int(llx[i]-option.cutImage[2])) + ',' + str(int(lly[i])) + ',' + str(int(urx[i]-option.cutImage[2])) + ',' + str(int(ury[i])) + ','
							entry = entry + str(option.class_map[flags[i]])
							entries.append(entry)
							continue

						# else if box is atleast 30 pixel in to the right half, include box
						elif(urx[i] >= option.cutImage[2]+30):
							entry = loc + vidname + '/' + "%03d.jpg," % f_num[i]
							entry = entry + str(int(0)) + ',' + str(int(lly[i])) + ',' + str(int(urx[i]-option.cutImage[2])) + ',' + str(int(ury[i])) + ','
							entry = entry + str(option.class_map[flags[i]])
							entries.append(entry)
							continue

					# if training with left half of image, the box co-ordinates do not need to be shifted
					elif option.name[-1] == 'L':

						# if box completely within split left half, include box
						if (urx[i] <= option.cutImage[3]):
							entry = loc + vidname + '/' + "%03d.jpg," % f_num[i]
							entry = entry + str(int(llx[i])) + ',' + str(int(lly[i])) + ',' + str(int(urx[i])) + ',' + str(int(ury[i])) + ','
							entry = entry + str(option.class_map[flags[i]])
							entries.append(entry)
							continue

						# else if box is atleast 30 pixel in to the left half, include box
						elif(llx[i] <= option.cutImage[3]-30):
							entry = loc + vidname + '/' + "%03d.jpg," % f_num[i]
							entry = entry + str(int(llx[i])) + ',' + str(int(lly[i])) + ',' + str(int(option.cutImage[3])) + ',' + str(int(ury[i])) + ','
							entry = entry + str(option.class_map[flags[i]])
							entries.append(entry)
							continue

					# training with whole images
					else:
						entry = loc + vidname + '/' + "%03d.jpg," % f_num[i]
						entry = entry + str(int(llx[i])) + ',' + str(int(lly[i])) + ',' + str(int(urx[i])) + ',' + str(int(ury[i])) + ','
						entry = entry + str(option.class_map[flags[i]])
						entries.append(entry)
						continue



print "\nGenerating trainset list ...\t",
dataset = open('trainset.txt', 'w')

for entry in entries:
	dataset.write("%s\n" % entry)

dataset.close()

b = []
for entry in entries:
	a = entry.strip().split(',')
	b.append(a[0])

option.num_frames = len(b)
b = set(b)


print "Done!\t",
print "Total Frames: %d" %len(b)


print "\nStarting Training Session"
print "----------------------------\n"
print "Trainset Settings : \n"
print "N = ",
print N
print "effects = ",
print effects
print "levels = ",
print levels
if syn and not only_syn:
	print "Real & Synthesized Videos"
elif syn and only_syn:
	print "Only Synthesized Videos"
elif not syn and not only_syn:
	print "Only Real Videos"

print "-----------------------------\n"

trainModel(option)
