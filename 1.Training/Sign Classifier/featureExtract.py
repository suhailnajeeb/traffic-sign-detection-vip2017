from scipy import misc
import numpy as np
import glob
import logging
import os
import threading
from Queue import Queue

# ----------------------------------- Settings ------------------------------

# Multi-threading Parameters (3 optimum)
max_threads = 8

# The images are RGB.
img_channels = 3

# Resize Parameters
img_rows = 64
img_cols = 64

# Data type = train/test
name = '...'
desc = 'Test Dataset comprised of ...'

# Video Selecti
# Set N_vid = [0,0] to process only train videos, and N_vid = [0,1] to process only test videos
# if you want to list particular videos, put N_vid[0] = -1 and the follow with video # for rest of the elements
# else if you want a particular range, just enter the range in N_vid; e.g [1,9] will process video # 1 to video # 9

N_vid = [0,0]
effects = 0,0]
levels =[0,0]
syn = 1
only_syn = 0

# Glob accumalator Arrays
X_temp=[]		# pixels
Y_temp=[]		# labels

# Directories
locR = '.../ROIs/.../Real/Cropped/'
locS = '.../ROIs/.../Synthesized/Cropped/'

if N_vid[0] is 0 and N_vid[1] is 0:
	typ = 'train'
elif N_vid[0] is 0 and N_vid[1] is 1:
	typ = 'test'
else:
	typ = 'train'

train_feat = '/' + name + '_' + typ + '_' + str(img_rows) + '_' + str(img_cols)

# ------------------------ Code ---------------------------------------

# test & train videos - given in the VIP Cup 2017 pdf
train_vid_rea =[1,2,3,9,10,11,12,13,14,15,16,17,20,22,23,25,27,28,29,30,32,33,34,35,36,37,40,42,43,44,45,46,48,49]
train_vid_syn = [1,3,5,7,8,10,11,14,15,19,21,23,24,25,26,27,29,30,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49]
test_vid_rea = [4,5,6,7,8,18,19,21,24,26,31,38,39,41,47]
test_vid_syn = [2,4,6,9,12,13,16,17,18,20,22,28,31,32,36]


# Queues
que_glob = Queue()
que_append = Queue()
k = 0

# function to make directories
def ensur_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

def getFeatures(q):
	global k
	global img_rows
	global img_cols
	global que_append

	while True:
		job = q.get()

		loc = job[0]
		vidname = job[1]

		a = sorted(glob.glob(loc + vidname + '/*.jpg'))
		l = len(a)

		flags = job[2]
		X_temp =[]
		Y_temp = []

		print " Processing:\t" + vidname

		for j in range (0,l):													# l = number of ROI in video
			roi = misc.imread(a[j])												# reading pixel
			roi_rsz = misc.imresize(roi,(img_rows,img_cols),interp='bicubic')	# resizing
			X_temp.append(roi_rsz)												# appending pixels
			Y_temp.append(flags[j])												# setting ytrain value


		print " Completed:\t" + vidname
		q.task_done()
		k += l
		pixels = [X_temp,Y_temp]
		que_append.put(pixels)

# function to apppend globbed pixels
def getGlobs(q):
	global X_temp
	global Y_temp

	while True:
		pixels = q.get()
		X_temp.extend(pixels[0])
		Y_temp.extend(pixels[1])
		q.task_done()


# creating directory
ensur_dir(train_feat)

# log file
logging.basicConfig(filename='/../Features/' + name + '_' + typ + '_' + str(img_rows) + '_' + str(img_cols) + '.log',level=logging.INFO,filemode='w')
logging.info("\tName: \t%s" % name)
logging.info("\tDescription: %s" % desc)
logging.info("----------------------------------")
logging.info("\tSize : \t\t%d x %d" % (img_rows,img_cols))
logging.info("\tN_vid = \t\t[%d,%d]" % (N_vid[0],N_vid[1]))
logging.info("\teffects = \t[%d,%d]" % (effects[0],effects[1]))
logging.info("\tlevels = \t[%d,%d]" % (levels[0],levels[1]))
logging.info("----------------------------------")
logging.info("\tSynthesized = \t%d" % syn)
logging.info("\tOnly Synthesized  = \t%d" % only_syn)
logging.info("----------------------------------")


# starting threads to glob
for i in range(max_threads-1):   	# only main thread gets to call other threads
        t = threading.Thread(target=getFeatures,args=(que_glob,))   # assigning target to thread
        t.setDaemon(True)                                       	# designating as Daemon Thread -> runs in bg
        t.start()                                              		# starting thread

# one thread to append globs
if __name__ == '__main__':
    ta = threading.Thread(target=getGlobs,args=(que_append,))
    ta.setDaemon(True)
    ta.start()

# queuing jobs
for t in range (1+only_syn,2+syn):

	if t == 1:
		loc = locR

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_rea
		elif N_vid[0] == 0 and N_vid[1] == 1:
			N = test_vid_rea
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[0],N_vid[1]+1)
	elif t==2 :
		loc = locS

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_syn
		elif N_vid[0] == 0 and N_vid[1] == 1:
			N = test_vid_syn
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[1],N_vid[2]+1)

	for n in N:
		try:
			label = np.genfromtxt(loc + "%02d_%02d_00_00_00/Ytrain.csv" % (t,n) , delimiter=',')
			l = len(label)
			flags = label[:,1]
		except:
			logging.info("\t%02d_%02d_00_00_00 has no ROI"%(t,n))
			continue

		if flags[0] == 0:
			logging.info("\t%02d_%02d_00_00_00 has no ROI"%(t,n))
			continue

		for eff in range(effects[0],effects[1]+1):
		
			if t == 2 and eff == 12:
				continue

			lvl = levels[0]

			while lvl < levels[1]+1:

				if eff == 0:
					vidname = "%02d_%02d_00_00_00" % (t,n)
					lvl = 10
				else:
					vidname = "%02d_%02d_01_%02d_%02d" % (t,n,eff,lvl)
					lvl += 1

				print " Queueing:\t" + vidname
				job = [loc,vidname,flags]
				que_glob.put(job)


# waiting for jobs to finish
que_glob.join()
que_append.join()

print "\n ---------------------------\n Globbing Complete!\n Converting to NumPy Array..."
X_train = np.array(X_temp)
Y_train = np.array(Y_temp)

print "\n Pixel Sets:\t",
print X_train.shape
print " Sign Sets:\t",
print Y_train.shape
print "\n %d ROIs processed" %k
logging.info(" --------------------------------")
logging.info("\t%d ROIs processed" %k)


# saving data set
# savez needs to be told names of array as well, <name=array>
print "\n Saving dataset to disk ..."
np.savez(train_feat, X_train=X_train,Y_train=Y_train)

print "\n Done!"
print "\n %5.1f MB file generated" %(os.path.getsize(train_feat + '.npz')/1000000.0)
