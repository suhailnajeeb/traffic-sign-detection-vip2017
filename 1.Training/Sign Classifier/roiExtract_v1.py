# This code gets the ROI from each frame of each video in the data set
# Change the location and destination of frames and ROI in the settings section
# Set max number of threads and number of video to process as well
# Recommended number of threads = 15 for core i7
# To process fames of Synthesized videos, set syn = 1

# Requires Frames to be in one folder, i.e all Real frames in a "Real" Folder
# and all Synthesized frames in a "Synthesized" Folder

import os
import threading
from Queue import Queue
import numpy as np
from numpy import random
import PIL
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

#----------------------------- Settings ---------------------------------------

max_threads = 8  			# number of maximum active threads at a given time

N_vid = [1,49] 				# number of videos, 1-49
effects = [1,12]			# number of effects, 0-12 implies all;
levels = [1,5]				# number of levels, 1-5
syn = 1 					# include Synthesized Videos
only_syn = 1				# make 1 to extract only from Synthesized videos

# dis_max =  0-1; max percentage of ROI dimensions to dipslace, 0 for no displace, 0.5 for 50% displacment
# dis_base = 0-1; 0.5 implies all displacments are atleast 0.5 * max displacement
dis_max = 0
dis_base = 0

# 2 implies ROI is enlarged to twice it's area; 0.5 implies ROI area is halved; None for no zoom in/out
zoom_factor = 0

# Directories
csv_dir = '.../labelscsv/'		# location of ROI labels
sav_dir = '.../ROIs/NoZoom/'		# save location for extracted ROIs
frame_dir = '.../Frames/' 		# location of extracted frames

#----------------------------- Function definitions ---------------------------

# function to make directories
def ensur_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

# makes sure x-coordinates are within frame after displace/zoom
def roundx(x):
	return 0 if x<0 else 1627 if x>1627 else x

# makes sure y-coordinates are within frame after displace/zoom
def roundy(y):
	return 0 if y<0 else 1235 if y>1235 else y

# function to displace ROI
def displaceROI(llx,lly,urx,ury,dis_max=dis_max,dis_base=dis_base):

	factor = random.random()+dis_base

	factor = 1 if factor>1 else factor

	dx = int(dis_max * factor * abs(urx-llx))
	dy = int(dis_max * factor * abs(ury-lly))

	direction = {1 : (llx,roundy(lly+dy),urx,roundy(ury+dy)),							# up
				 2 : (llx,roundy(lly-dy),urx,roundy(ury-dy)),							# down
				 3 : (roundx(llx+dx),lly,roundx(urx+dx),ury),							# right
				 4 : (roundx(llx-dx),lly,roundx(urx-dx),ury),							# left
				 5 : (roundx(llx+dx),roundy(lly+dy),roundx(urx+dx),roundy(ury+dy)),		# up-right
				 6 : (roundx(llx+dx),roundy(lly-dy),roundx(urx+dx),roundy(ury-dy)),		# down-right
 				 7 : (roundx(llx-dx),roundy(lly+dy),roundx(urx-dx),roundy(ury+dy)),		# up-left
 				 8 : (roundx(llx-dx),roundy(lly-dy),roundx(urx-dx),roundy(ury-dy))}		# down-left

	i = random.randint(1,9)	# random int from 1 - 9

	return direction[i]


def zoomROI(llx,lly,urx,ury,zoom_factor=zoom_factor):

	# width,height of ROI
	x = abs(urx-llx)
	y = abs(ury-lly)

	# calculating amount to move each corner of ROI
	dx = np.round((zoom_factor*x - x)/2.0)
	dy = np.round((zoom_factor*y - y)/2.0)

	return roundx(llx-dx),roundy(lly-dy),roundx(urx+dx),roundy(ury+dy)


def getROI(q):

	while True:
		vid = q.get()
		vid_name = vid[0]
		dataset  = vid[1]
		c_vid_fldr = vid[2]
		img_dir = vid[3]

		try:
			l = len(dataset)		 # number of entries in csv
			llx = dataset[:,2]
			lly = dataset[:,3]
			urx = dataset[:,8]
			ury = dataset[:,9]
			f_num = dataset[:,0]	# frame number
			flags = dataset[:,1]	# flag type
		except:
			s = (1,2)
			Ytrain = np.zeros(s)												# empty Ytrain
			np.savetxt(c_vid_fldr + '/Ytrain.csv' ,Ytrain,delimiter=",")		# saving Ytrain in with resized images
			print "Has No ROI:\t" + vid_name
			q.task_done()
			continue

		prv_f = 301		# used to handle duplicate frames, random value for first iteration
		dup = 1

		s = (l-1,2)				 # size of Ytrain,l-1 rows, since first row = headers
		Ytrain = np.zeros(s)	 # Ytrain has labels of flag type for frames

		print "Processing:\t" + vid_name

		for i in range(1,l):
			try:
				img = Image.open(img_dir + vid_name + "/%03d.jpg" % f_num[i])
			except IOError:
				print "Error Opening:\t" + vid_name + "\t%03d.jpg" % f_num[i]
				continue

			# actual corner co-ordinates
			x1,y1,x2,y2 = llx[i],lly[i],urx[i],ury[i]

			# zooming out/in ROI
			if zoom_factor is not None:
				x1,y1,x2,y2 = zoomROI(x1,y1,x2,y2)			# zoomout ROI by factor

			# displacing
			if dis_max>0:
				x1,y1,x2,y2 = displaceROI(x1,y1,x2,y2)		# displace ROI randomly

			# cropping
			img_crop = img.crop((x1,y1,x2,y2))

			#saving
			if prv_f!=f_num[i]:
				dup = 1			#reset duplicate index
			else:
				dup = dup + 1	#increment duplicate index

			img_crop.save((c_vid_fldr + '/%03d_%02d.jpg'% (f_num[i],dup)))

			#setting Y train value
			Ytrain[i-1,0]= "%03d" % (f_num[i])		#frame number
			Ytrain[i-1,1]= flags[i]					#flag number

			prv_f=f_num[i]		# used to check for duplicate frames, so as to not overwrite while saving

		np.savetxt(c_vid_fldr + '/Ytrain.csv' ,Ytrain,delimiter=",")		#saving Ytrain in with resized images
		print "Completed:\t" + vid_name
		q.task_done()


#--------------------------------- Main Code -----------------------------------

# list of train and test videos
train_vid_rea =[1,2,3,9,10,11,12,13,14,15,16,17,20,22,23,25,27,28,29,30,32,33,34,35,36,37,40,42,43,44,45,46,48,49]
train_vid_syn = [1,3,5,7,8,10,11,14,15,19,21,23,24,25,26,27,29,30,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49]
test_vid_rea = [4,5,6,7,8,18,19,21,24,26,31,38,39,41,47]
test_vid_syn = [2,4,6,9,12,13,16,17,18,20,22,28,31,32,36]

que = Queue()	# lists videos and keeps then in queue
vid_info = []	# list contains frame info given to threads

if dis_max>0 and zoom_factor is not None:
	sav_dir = sav_dir + 'ZoomedOut_' + str(zoom_factor) + 'Displaced_' + str(dis_max) + '_' + str(dis_base) + '/'
elif dis_max>0:
	sav_dir = sav_dir + 'Displaced_' + str(dis_max) + '_' + str(dis_base) + '/'
elif zoom_factor is not None:
	sav_dir = sav_dir + 'ZoomedOut_' + str(zoom_factor) + '/'


# creating directories
ensur_dir(sav_dir)

# starting threads
for c in range(max_threads):
    if __name__ == '__main__':                               # only main thread gets to call other threads
        t = threading.Thread(target=getROI,args=(que,))      # assigning target to thread
        t.setDaemon(True)                                    # designating as Daemon Thread -> runs in bg
        t.start()                                            # starting thread

# getting labels and creating sub directories
for v in range(1+only_syn,2+syn):

	if v == 1:
		vidtype = "Real"

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_rea
		elif N_vid[0] == 0 and N_vid[1] == 1:
			N = test_vid_rea
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[0],N_vid[1]+1)
	elif v==2 :
		vidtype = "Synthesized"

		if N_vid[0] == 0 and N_vid[1] == 0:
			N = train_vid_syn
		elif N_vid[0] == 0 and N_vid[1] == 1:
			N = test_vid_syn
		elif N_vid[0] == -1:
			N = N_vid[1:]
		elif N_vid[0]>0:
			N = range(N_vid[0],N_vid[1]+1)

	frame_dir2 = frame_dir + vidtype + "/"

	for n in N:
		label = csv_dir + "%02d_%02d.csv"% (v,n)	# loading labels
		data = np.genfromtxt(label,delimiter=',')

		try:
			Length = len(data)
		except:
			print "%02d_%02d has no ROI" % (v,n)
			continue

		for eff in range(effects[0],effects[1]+1):

			if eff == 12 and v == 2:
				continue

			lvl = levels[0]

			while lvl<levels[1]+1:
				if eff == 0:		# no effect, then set this name and ...
					vid_name = "%02d_%02d_00_00_00" % (v,n)
					lvl = 10		# ... go to end of lvl loop
				else:
					vid_name = "%02d_%02d_01_%02d_%02d" % (v,n,eff,lvl)
					lvl+=1

				c_fldr = sav_dir + "/" + vidtype + "/Cropped/" + vid_name +"/"		# creating individual folders

				ensur_dir(c_fldr)	# creating sub directories

				vid_info = [vid_name,data,c_fldr,frame_dir2]
				que.put(vid_info)

que.join()           # blocks main until dameon threads have finished their job
print "Done !"

#------------------------------------------------------------------------------
