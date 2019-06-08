import os
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import threading
from Queue import Queue

#Real: N = [1,2,3,9,10,11,12,13,14,15,16,17,20,22,23,25,27,28,29,30,32,33]
#Synthesized: N = [1,3,5,7,8,11,14,15,19,21,25,29,35,38,42,45,47,49]


csv_dir = '.../csvTrained/Synthesized/'
sav_dir = '..../ManualTrain/Synthesized/Cropped/'
frame_dir = '.../Frames/Synthesized/' 

V = [2]
N = [1,3,5,7,8,11,14,15,19,21,25,29,35,38,42,45,47,49]

E = range(1,13)
L = range(1,6)

max_threads = 8

que_vid = Queue()

def main():
	for i in range(max_threads):   						# only main thread gets to call other threads
	        t = threading.Thread(target=extractROI,args=(que_vid,))   	# assigning target to thread
	        t.setDaemon(True)                                       	# designating as Daemon Thread -> runs in bg
	        t.start() 

	for v in V:
		for n in N:
			for eff in E:
				if ((v==2) and (eff == 12)): continue
				for lvl in L:
					if not (eff==0): typ = 1 
					else: typ = 0
					vid_name = "%02d_%02d_%02d_%02d_%02d" % (v,n,typ,eff,lvl)
					print "Queueing:\t"+vid_name
					que_vid.put([vid_name,v,n])				

def ensur_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

def extractROI(q):

	while True:
		
		
		job = q.get()
		vid_name = job[0]
		v = job[1]
		n = job[2]

		print "Processing:\t"+vid_name

		label = csv_dir + "%02d_%02d_train.csv"% (v,n)
		c_vid_fldr = sav_dir + vid_name
		ensur_dir(c_vid_fldr)
		dataset = np.genfromtxt(label,delimiter=',')

		try:
			Length = len(dataset)
		except:
			print "%02d_%02d has no ROI" % (v,n)
			#continue

		l = len(dataset)	 # number of entries in csv
		llx = dataset[:,1]
		lly = dataset[:,2]
		urx = dataset[:,3]
		ury = dataset[:,4]
		f_num = dataset[:,0]	# frame number
		flags = dataset[:,5]

		prv_f = 301
		dup = 1
		s = (l-1,2)
		Ytrain = np.zeros(s)


		for i in range (1,l):
		
			try:
				img = Image.open(frame_dir + vid_name + "/%03d.jpg" % f_num[i])
			except IOError:
				print "Error Opening:\t" + vid_name + "\t%03d.jpg" % f_num[i]
				continue
			x1,y1,x2,y2 = llx[i],lly[i],urx[i],ury[i]
			img_crop = img.crop((x1,y1,x2,y2))
		
			if prv_f!=f_num[i]:
				dup = 1	
			else:
				dup = dup + 1

			img_crop.save((c_vid_fldr + '/%03d_%02d.jpg'% (f_num[i],dup)))

			Ytrain[i-1,0]= "%03d" % (f_num[i])
			Ytrain[i-1,1]= flags[i]

			prv_f=f_num[i]

		np.savetxt(c_vid_fldr + '/Ytrain.csv' ,Ytrain,delimiter=",")
		print "Completed:\t" + vid_name
		q.task_done()

main()
que_vid.join()
				
