from frameExtractLib import *
from frcnn.frcnnLib import *
from cnn.cnnLib import *
from tracker.hybridTrack2 import *
from options import*

import cv2
import threading
from Queue import Queue
import time
import numpy as np
from optparse import OptionParser
import time
from progressbar import ProgressBar as pb
import os

# suprresing tensorflow messages
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
configTF = tf.ConfigProto()
configTF.gpu_options.allow_growth = True
sess = tf.Session(config=configTF)



# this function formats the results from the model to generate the files in detection.zip
def formatResults(f,idx,bbox,signClass):

	idx = "%03d"%idx
	out = [ (box,sign) for box,sign in zip(bbox,signClass) if (sign >0 and sign <15) ]

	for box,sign in out:
		ulx,uly,lrx,lry = box

		# boxes obtained from tracker module sometimes cross frame edge; correcting for that here
		ulx = 0 if ulx<0 else 1627 if ulx>1627 else ulx
		uly = 0 if uly<0 else 1627 if uly>1627 else uly
		lrx = 0 if lrx<0 else 1627 if lrx>1627 else lrx
		lry = 0 if lry<0 else 1627 if lry>1627 else lry

		# assigning the other corner co-ordinates
		llx,ury,urx,lly = ulx,uly,lrx,lry

		line = '_'.join(map(str,[idx,sign,llx,lly,lrx,lry,ulx,uly,urx,ury]))

		# writing to file
		f.write(line + '\n')


# this fucntion will populate queues with images for different parts of the system
def populateQ(frames,imgQ,imgVisualizeQ,frcnnQL,frcnnQR,LKTrackQ,chTrackQ,visualize,):

	# overlap between two halves of the frcnn input images
	overlap = 70

	# index to keep track of frames
	idx= 0
	while True:
		try:
			idx +=1

			ch_type,img = frames.next()										# getting frame and challenge type

			if idx % 7 ==0:
				chTrackQ.put(ch_type)										# Putting challenge type in queue every 7 frames to be used later when determining classifier model to use

			imgQ.put(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))					# Queue for Sign Classifier (RGB Images)

			if visualize:
				imgVisualizeQ.put(img)										# Queue for visualizing boudning boxes (BGR image if using cv2.imshow())

			frcnnQL.put(img[0:660,0:814+overlap/2])							# Bounding box detection Queue (left side ,upper half of frame , BGR Images)
			frcnnQR.put(img[0:660,814-overlap/2:1627])						# Bounding box detection Queue (right side, upper half of frame, BGR Images)
			LKTrackQ.put([ch_type,cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)])	# LK Tracker Queue (Gray Images)

			if idx == 300:													# 300 frames queued, reset index
				idx = 0

		except StopIteration:
			return

def main():
	try:
		parser = OptionParser()

		parser.add_option("-p", "--vid_path", dest="vid_fldr", help="Path to Videos to test.",default='../../../../data/Videos/All/')
		parser.add_option("--fp", "--frame_path", dest="frame_fldr", help="Path to Frames to test",default='../../../../Frames/Real/'),
		parser.add_option("-f", "--frame_gen", dest="frame_gen",help="Frame Generator Selection; if flag included, will generate frames from extracted frames", action= 'store_false', default=True)
		parser.add_option("--lp", "--label_path", dest="label_path", help="Path to GT labels (for visualization)", default='./labels/'),
		parser.add_option("-r", "--results", dest="generateCSV",help="Flag. Produces detection text files", action= 'store_false', default=True)
		parser.add_option("-v", "--visualize", dest="visualize", help="Flag. Shows preview of video and detections",action= 'store_true', default=False)
		parser.add_option("--vb", "--verbose", dest="verbose", help="Flag. Prints outputs from FRCNN,Tracker and Classifier",action= 'store_true', default=False)

		parser.add_option("-n", "--vid", dest="vidname", help=" Name of videos to test (if testing specific videos)" ,default=None)

		parser.add_option("--scaleBoxes", dest="scaleBoxes", help=" float : 0.0 -> 1.0 \n Amount boxes are scaled after detection ", default=1.0)
		parser.add_option("--showROI", dest="showROI", help="Bool. Preview detected regions in seperate window", action= 'store_true', default=False)
		parser.add_option("--part", dest="test_part", help="Generate results for part of test dataset\n  --part <integer 1-6>" ,default=None)
		parser.add_option("--continue", dest="continue_last", help="Continues result generation from video last processed; used in case code terminates prematurely",action= 'store_true' ,default=False)

		(options, args) = parser.parse_args()

		# path to video/frames
		vid_fldr = options.vid_fldr
		frame_fldr = options.frame_fldr

		if options.frame_gen and not os.path.exists(vid_fldr):
			raise Exception("Error: video directory does not exist")
		elif not options.frame_gen and not os.path.exists(frame_fldr):
			raise Exception("Error: frame directory does not exist")

		# path to labels (for visualization)
		gtlabels = options.label_path

		# flags
		verbose = options.verbose
		visualize = options.visualize
		generateCSV = options.generateCSV

		# maximum videos frames to hold in queue
		max_vid_que = 1

		# Queues (will hold 1200 frames ~ 4 videos at max.
		# Will start filling again once consumed by model)
		imgQ = Queue(maxsize = 300*max_vid_que)						# Queue for Images to be used by Sign Classifier
		imgVisualizeQ = Queue(maxsize =  300*max_vid_que)			# Queue for Images to be used by visualization code
		frcnnQL = Queue(maxsize =  300*max_vid_que)					# Queue for Images to be used by FRCNN (bounding box detector), left half of image
		frcnnQR = Queue(maxsize =  300*max_vid_que)					# Queue for Images to be used by FRCNN (bounding box detector), right half of image
		LKTrackQ = Queue(maxsize =  300*max_vid_que)
																	# Queue for Images to be used by box tracker system (Lucas Kanade/Optical Flow)
		cnnQ = Queue(maxsize =  300*max_vid_que)					# Queue for bounding boxes to be used by Sign Classifier (will get co-ordinates and crop accodingly)
		bboxQ = Queue()												# Queue for bounding boxes to be used by tracker system
		trackerOutQ = Queue()										# Queue for bounding boxes output from tracker system (used for visualization)
		cnnOutQ = Queue()											# Queue for classes output from sign Classifier (used for feedback in tracker system)
		chTrackQ = Queue()											# Queue for keeping track of challenge types (used to dyanmically change classifier model weights)

		# Initializing settings for neural networks
		frcnnSettings =  optionsFRCNN()
		cnnSettings = optionsCNN()

		print "\nInitializing Tracker... ",
		tracker = hybridTracker()
		print "Complete."

		print "\nInitializing FRCNN ... ",
		C,frcnnRPN,frcnnClass,frcnnClassOnly = setupFRCNN(frcnnSettings)
		print "Complete."

		print "\nBuilding Neural Networks ...",
		cnn = createModel(cnnSettings)
		loadWeights(cnn,cnnSettings.model_weights_default)
		print "Complete."

		test_vid_rea = [4,5,6,7,8,18,19,21,24,26,31,38,39,41,47]		# test video sequences (real videos)
		test_vid_syn = [2,4,6,9,12,13,16,17,18,20,22,28,31,32,36]		# test video sequences (synthesized videos)

		# When we generated the result, we split the test videos into 6 parts and ran it on 6 PCs
		if options.test_part is not None:
			if int(options.test_part) == 1:
				test_vid_rea = test_vid_rea[:5]				# first 5 real video sequences
				test_vid_syn = []
			elif int(options.test_part) == 2:
				test_vid_rea = test_vid_rea[5:10]			# second 5 real video sequences
				test_vid_syn = []
			elif int(options.test_part) == 3:
				test_vid_rea = test_vid_rea[10:]			# third 5 real video sequences
				test_vid_syn = []
			elif int(options.test_part) == 4:
				test_vid_rea = []
				test_vid_syn = test_vid_syn[:5]				# first 5 synthesized video sequences
			elif int(options.test_part) == 5:
				test_vid_rea = []
				test_vid_syn = test_vid_syn[5:10]			# second 5 synthesized video sequences
			elif int(options.test_part) == 6:
				test_vid_rea = []
				test_vid_syn = test_vid_syn[10:]			# third 5 synthesized video sequences


		# if testing individual videos, command line argument used
		if options.vidname is not None:
			vidname = options.vidname.split(",")

		# else all videos in test dataset processed
		else:
			vidname = []
			vidname.extend( ["01_%02d_00_00_00" %(seq) for seq in test_vid_rea]	)																# Real No Challenge
			vidname.extend( ["02_%02d_00_00_00" %(seq) for seq in test_vid_syn] )																# Syn No Challenge
			vidname.extend( ["01_%02d_01_%02d_%02d" %(seq,eff,lvl) for seq in test_vid_rea for eff in range(1,13) for lvl in range(1,6)] )		# Real Challenge
			vidname.extend( ["02_%02d_01_%02d_%02d" %(seq,eff,lvl) for seq in test_vid_syn for eff in range(1,12) for lvl in range(1,6)] )		# Syn Challenge

			print "\nGenerating Detections for all test videos (%d videos):" % len(vidname)
			print "-----------------------------------------------------------"

		# if --continue flag given, we skip videos for which detection file already generated
		if options.continue_last:
			adjusted_vidname = []
			print "\n"
			for vid in vidname:
				detection_file = './detections/' + vid + '.txt'

				if os.path.exists(detection_file):
					print "Skipping %s: Detection file already exists" %vid
					continue
				else:
					adjusted_vidname.append(vid)

			vidname = adjusted_vidname

		# frameGenerator 1 gets frames from video, 2 from extracted frames
		if options.frame_gen:
			frames = frameGenerator1(vidname,vid_fldr,classify=True)
		else:
			frames = frameGenerator2(vidname,frame_fldr,classify=True)

		# need to iterate generator once before handing over to another thread
		# this is because of peculiar way keras's  model.predict() behaves when multi-threading
		overlap = 70
		ch_type,img = frames.next()
		imgQ.put(img)
		imgVisualizeQ.put(img)
		frcnnQL.put(img[0:660,0:814+overlap/2])
		frcnnQR.put(img[0:660,814-overlap/2:1627])
		LKTrackQ.put([ch_type,cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)])

		# thread to feed get frames,preprocess and feed queues
		t1 = threading.Thread(target=populateQ,args=(frames,imgQ,imgVisualizeQ,frcnnQL,frcnnQR,LKTrackQ,chTrackQ,visualize,))
		t1.setDaemon(True)
		t1.start()

		# thread for tracker module
		t2 = threading.Thread(target=tracker.track,args=(LKTrackQ,cnnOutQ,bboxQ,cnnQ,trackerOutQ,))
		t2.setDaemon(True)
		t2.start()

		# Setting up generator functions
		Right = findBBox(frcnnQR,'R',C,frcnnSettings,frcnnRPN,frcnnClassOnly)				# bounding box detector (right side of frames)
		Left = findBBox(frcnnQL,'L',C,frcnnSettings,frcnnRPN,frcnnClassOnly)				# bounding box decetor  (left side of frames)
		Classify = classifySign(cnn,cnnQ,cnnOutQ,imgQ,cnnSettings,class_map=None,scale=float(options.scaleBoxes),showROI=options.showROI)		# sign classifier


		for vid in vidname:
			try:
				print "\n\n---------------------------------"
				print "Processing:\t %s" % vid
				print "---------------------------------\n"

				if not verbose:
					bar = pb(max_value = 300)		# progress bar

				# loading labels to plot ground truth
				if visualize:
					try:
						labels = np.genfromtxt(gtlabels + vid[:5] +'.txt',delimiter='_')
						fnum = labels[:,0]
						gtbox = np.hstack((labels[:,2:4],labels[:,8:10]))
					except:
						print "\nError Loading Labels: please check label folder directory"
						print "Continuing without visualization\n"
						visualize = False
						pass

				# creating files to store detections
				if generateCSV:
					if not os.path.exists('./detections/'):
						os.makedirs('./detections/')

					detection_file = './detections/' + vid + '.txt'

					f = open( detection_file,'w')
					f.write("frameNumber_signType_llx_lly_lrx_lry_ulx_uly_urx_ury\n")


				# wait for queue to populate
				while frcnnQR.empty():
					continue

				idx = 0		# frame index

				# keeping track of challenge type/enviroment conditions
				prev_chType = 'NoCh'

				while idx<300:

					idx +=1

					# checking challenge type every 7 frames and changing classifier model weights for particular challenget type
					if idx % 7 == 0:
						curr_chType = chTrackQ.get()

						if prev_chType != curr_chType:
							loadWeights(cnn,cnnSettings.model_weights_effects[curr_chType])

							if verbose:
								print "\n---------------------------------------"
								print "Changing Classifier Model to : %s " % curr_chType
								print "---------------------------------------\n"

						prev_chType = curr_chType

					# bbox array will contain bounding boxes detected by FRCNN
					bbox = []

					# combining co-ordinates of boxes from left and right side of image
					bbox.extend(Left.next())
					bbox.extend(Right.next())

					# passsing boxex to tracker module through queue
					bboxQ.put(bbox)

					# classifier gets boxes from tracker module through another queue and returns box classes
					y = Classify.next()

					# boxes forwared to classifier by tracker (used for visualization and writing to file)
					bboxT = trackerOutQ.get().tolist()
					trackerOutQ.task_done()

					if verbose:
						print "Frame %03d: BBoxes :\t"%idx,
						print bboxT
						print "            Class :\t",
						print y
						print"\n"

					if visualize:
						img = imgVisualizeQ.get()
						overlay = img.copy()
						cv2.rectangle(overlay,(5,5),(330,70),(10,10,10),-1)
						cv2.putText(overlay,"Frame : %03d"%idx ,org = (10,50),fontFace =cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,20),thickness=4,fontScale=1.5)

						# plotting Tracker output (Green)
						for sign,box in enumerate(bboxT):
							try:
								box = scaleBoxes(box,float(options.scaleBoxes))
								cv2.rectangle(overlay,(box[0],box[1]),(box[2],box[3]),(0,255,0),4)
								if y is not []:
									cv2.rectangle(overlay,(box[0]-10,box[3]+38),(box[0]+138,box[3]+3),(10,10,10),-1)
									cv2.putText(overlay,"Class: %d"%(y[sign]),org=(box[0],box[3]+30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,20),thickness=2,fontScale=0.9)
							except:
								continue


						# plotting GT boxes (Blue)
						try:
							for box in np.int16(gtbox[np.where(fnum==idx)]):
								box = scaleBoxes(box,float(options.scaleBoxes))
								cv2.rectangle(overlay,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
						except:
							pass

						cv2.addWeighted(overlay,0.7,img,0.3,0,img)

						img = cv2.resize(img,(814,618),interpolation=cv2.INTER_CUBIC)
						cv2.imshow('img',img)
						cv2.waitKey(1)
						imgVisualizeQ.task_done()

					if generateCSV:
						formatResults(f,idx,bboxT,y)		# writing to file

					if not verbose:
						bar.update(idx)						# updating progress bar

					if idx == 300:
						break

				if generateCSV:
					f.close()

			except StopIteration:
				if generateCSV:
					f.close()

	except Exception as e:
		print "\nExiting : %s \n" %e
		if generateCSV:
			f.close()
			os.remove(detection_file)		# removing last opened file

	except KeyboardInterrupt:
		print "\n---------------\nUser Stopped the Program.\n"
		if generateCSV:
			f.close()
			os.remove(detection_file)		# removing last opened file

if __name__=="__main__":
	main()
