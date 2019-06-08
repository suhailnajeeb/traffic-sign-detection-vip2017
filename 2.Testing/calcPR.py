import numpy as np
import matplotlib.pyplot as pyplot
import glob
from optparse import OptionParser
from progressbar import ProgressBar
import matplotlib.pyplot as plt

'''
This code is used to calculate the number of True Positives (TP), False Positive (FP),
True Negative (TN) and False Negative (FN). We expand upon the definitions of these
metrics given in the VIP Cup 2017 description.
------------------------------------------------------------------------------------------------------------------------------------------

True Positives 	: 	the number of OT boxes with sign type correctly identified and at least 50% overlap with the ground truth annotation

------------------------------------------------------------------------------------------------------------------------------------------

True Negatives 	:	the number of frames not included in both GT labels and OT detection file

------------------------------------------------------------------------------------------------------------------------------------------

False Positives :

	A: the number of boxes in frames included in OutpuT (OT) detection file but not in Ground Truth (OT) labels

	B: the number of OT boxes with less than 50% overlap with GT box in Frames included in both GT labels and OT detection file

	C: the number of boxes with sign type incorrectly identified in OT detection file even though there is atleast 50% overlap in Frames included in both GT labels and OT detection file

------------------------------------------------------------------------------------------------------------------------------------------
False Negatives :

	A: the number of GT boxes in frames included in GT labels but not in OT detection file

	B: difference between number of GT boxes and OT boxes when # of GT boxes > OT boxes ; this implies system failed to detect that many GT boxes
------------------------------------------------------------------------------------------------------------------------------------------
'''

detection_path = './detections/'
label_path = './labels/'

# takes (x1,y1,x2,y2) where x2>x1 and y2>y1
# gives percetnage overalp of bi with ai
# if perfect overlap or bi box encompasses ai, returns 1.0
def calc_overlap(ai, bi):
	ai = np.int32(ai)
	bi = np.int32(bi)
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y


	if w < 0 or h < 0:
		return 0

	return w*h/((ai[2]-ai[0])*(ai[3]-ai[1])*1.0)

# calculating False Pos/Negs nad True Pos/Negs
def getPR(vidname,detection_path,label_path,verbose=False):

	GT = {}         # Ground Truth
	OT = {}         # OuTput from system

	tp,fp,tn,fn = 0,0,0,0		# true positive, false positive, true negative and false negative

	fpA,fpB,fpC = 0,0,0
	fnA,fnB = 0,0

	# loading dectection files and ground truth labels
	detection = np.genfromtxt(detection_path + vidname + '.txt',delimiter='_')
	groundTruth = np.genfromtxt(label_path + vidname[:5] + '.txt',delimiter='_')

	fnum = np.int16(detection[1:,0])					# frame numbers
	sign = np.int16(detection[1:,1])					# sign types
	box = np.int16(detection[1:,[6,7,4,5]])				# (ulx,uly,lrx,lry)

	fnumGT = np.int16(groundTruth[1:,0])
	signGT = np.int16(groundTruth[1:,1])
	boxGT = np.int16(groundTruth[1:,[2,3,8,9]])			# the columns in the original labels are mislabelled, these columns correspond to (ulx,uly,lrx,lry)

	# Getting data into dicts for ease of processing
	for idx,frame in enumerate(fnum):
		if frame not in OT:
			OT[frame] = {}
			OT[frame]['boxes'] = []
			OT[frame]['signs'] = []

		OT[frame]['boxes'].append(box[idx])
		OT[frame]['signs'].append(sign[idx])


	for idx,frame in enumerate(fnumGT):
		if frame not in GT:
			GT[frame] = {}
			GT[frame]['boxes'] = []
			GT[frame]['signs'] = []

		GT[frame]['boxes'].append(boxGT[idx])
		GT[frame]['signs'].append(signGT[idx])


	# Calculating results

	# False Negatives (A) -> Frames in GT but not in OT, with each frame contributing the number of GT boxes in that frame to False Negatives
	fnA = sum( [ len(GT[frame]['signs']) for frame in GT if frame not in OT ] )

	# True Negatives -> Frames not in GT and not in OT
	tn = len ( [b for b in range(1,301) if (b not in fnum) and (b not in fnumGT)] )

	# False Positives (A) -> Frames not in GT but in OT, with each frame contributing the number of Output boxes in that frame to False Positives
	fpA = sum ( [len(OT[frame]['signs']) for frame in OT if frame not in GT ] )


	# Now checking frames present in both OT and GT
	for frame in OT:
		if frame in GT:

			boxesOT = OT[frame]['boxes']
			boxesGT = GT[frame]['boxes']
			signsOT = OT[frame]['signs']
			signsGT = GT[frame]['signs']

			# if number of ground truth boxes greater than
			# number of output boxes, the difference is the number of False Negatives (B)
			if len(boxesGT) > len(boxesOT):
				fnB += len(boxesGT) - len(boxesOT)


			# comparing overlap of each output box with each ground truth box
			# overlap is a 2-D array where each row represents the overlap
			# of a ouptut box with the different groud truth boxes (columns)
			overlap = np.array(	[calc_overlap(boxGT,boxOT)	for boxOT in boxesOT for boxGT in boxesGT]	).reshape(-1,len(boxesGT))

			# converting to percentages
			overlap = np.int16( overlap * 100)

			if verbose:
				print"\n\n----%d-----\n"%frame
				print overlap
				print "\n"

			# returns row index (corresponds to output box) and col index ( corresponds to gt box) where overlap > or = 50 %
			row,col = np.where(overlap >= 50)

			# difference between number of output boxes and length of 'row'
			# (which is equal to number of boxes with overlap >50) corresponds to the
			# number of False Positives (B) for the frame
			fpB += len(boxesOT) - len(row)

			# now checking sign types of boxes that overlap more than 50 %
			for ot_idx,gt_idx in zip(row,col):

				# if sign types match, True Positive
				if signsOT[ot_idx] == signsGT[gt_idx]:
					tp +=1

				# else False Positive (C)
				else:
					fpC += 1
			if verbose:
				print "fpB: %d\nfnB: %d"%(len(boxesOT) - len(row),len(boxesGT) - len(boxesOT))
				print"\n------------\n\n"

	fp = fpA + fpB + fpC
	fn = fnA + fnB


	acc = 100*(tp+tn)/((tp+tn+fn+fp)*1.0)
	pre = 100*tp/((tp+fp)*1.0)
	rec = 100*tp/((tp+fn)*1.0)

	return [tp,fp,tn,fn,acc,pre,rec,fpA,fpB,fpC,fnA,fnB]


def main():

	parser = OptionParser()

	parser.add_option("-n", "--vidname", dest="vidname", help="\nEnter name of video to calculate P-R for. \nAlternatively enter 0 to calculate for all videos in ./detections",default='0')
	parser.add_option("--dp","--detection_path", dest="detection_path", help="\nPath to detection files", default = './detections/')
	parser.add_option("--lp","--label_path", dest="label_path", help="\nPath to ground truth labelfiles", default = './labels/')
	parser.add_option("-s","--save", dest="save", help="\nSave results to csv file", action = 'store_true', default = False)
	parser.add_option("-v","--verbose", dest="verbose", help="\nPrints breakdowns of FP and FN", action = 'store_true', default = False)

	(options, args) = parser.parse_args()

	if options.vidname == '0':
		try:
			vids = sorted(glob.glob(options.detection_path + '*.txt'))
			vids = [name[:-4] for _,_,name in (vid.split('/') for vid in vids)]

			print "\n"
			prog = ProgressBar(max_value=len(vids))
			results = []
			vid2 = []
			for i,vid in enumerate(vids):
				try:
					results.append(getPR(vid,options.detection_path,options.label_path))
					vid2.append(vid)
					prog.update(i)
				except Exception as e:
					print "\n\n\n %s : \t %s \n\n\n" %(vid,e)
					continue

			if options.save:

				with open('results.csv','w') as F:
					F.write("Seq,Effect,Level,TP,FP,TN,FN,Accuracy,Precision,Recall,FPa,FPb,FPc,FNa,FNb\n")
					for name,res in zip(vid2,results):
						res = "".join(str(res)[1:-1])
						name = name.split('_')
						seq = name[0]+'_'+name[1]
						eff = name[3]
						lvl = name[4]
						F.write("%s,%s,%s,%s\n" % (seq,eff,lvl,res))

			# results.sort(key=lambda x:x[1])
			# Precision = [res[0] for res in results ]
			# Recall = [res[1] for res in results ]
			# plt.figure()
			# plt.plot(Recall,Precision)
			# plt.show()

			tp,fp,tn,fn,_,_,_,fpA,fpB,fpC,fnA,fnB = sum(np.array(results))
			print "\n\n-------------------------"
			print "TP: %d\nFP: %d\nTN: %d\nFN: %d"%(tp,fp,tn,fn)
			print "-------------------------"
			print "Precision:\t %2.2f %%" %(100*(tp/((tp+fp)*1.0)))
			print "Recall:\t\t %2.2f %%" %(100*(tp/((tp+fn)*1.0)))
			print "-------------------------"
			print "Accuracy:\t %2.2f %%" %(100*(tp+tn)/((tp+tn+fp+fn)*1.0))
			print "-------------------------\n"

			if options.verbose:
				print "-------------------------"
				print "FpA: %d\nFpB: %d\nFpC: %d\nFnA: %d\nFnB: %d" %(fpA,fpB,fpC,fnA,fnB)
				print "-------------------------\n\n"

		except Exception as e:
			print e


	else:
		try:
			tp,fp,tn,fn,_,_,_,fpA,fpB,fpC,fnA,fnB = getPR(options.vidname,options.detection_path,options.label_path,options.verbose)
			print "\n-------------------------"
			print "TP: %d\nFP: %d\nTN: %d\nFN: %d"%(tp,fp,tn,fn)
			print "-------------------------"
			print "Precision:\t %2.2f %%" %(100*(tp/((tp+fp)*1.0)))
			print "Recall:\t\t %2.2f %%" %(100*(tp/((tp+fn)*1.0)))
			print "-------------------------"
			print "Accuracy:\t %2.2f %%" %(100*(tp+tn)/((tp+tn+fp+fn)*1.0))
			print "-------------------------\n"

			if options.verbose:
				print "-------------------------"
				print "FpA: %d\nFpB: %d\nFpC: %d\nFnA: %d\nFnB: %d" %(fpA,fpB,fpC,fnA,fnB)
				print "-------------------------\n\n"

		except Exception as e:
			print e

if __name__ == '__main__':
	main()
