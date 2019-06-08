# This code gets frames from each of the videos of the data set
# Change the location and destination of videos and frames in the settings section
# Set max number of threads and number of video to process as well
# Recommended number of threads = 8 for core i7 4core/8thread processor
# To process Synthesized videos, set syn = 1

# Requires Videos to be in one folder, i.e all Real videos in a "Real" Folder
# and all Synthesized videos in a "Synthesized" Folder

import threading
from Queue import Queue
import os
import av
import numpy
import logging

logging.basicConfig(filename='.../Frames/FrameExtraction.log',level=logging.DEBUG)

#--------------------------------- Settings ------------------------------------

max_threads = 8 # number of maximum active threads at a given time

N = [1,49]              # number of videos to process , 1-49
effects = [0,0]         # effect types 0-12 -> [0,0] implies no challenge
levels =  [0,0]         # effect levels 1-5
syn = 1                # set to 1 to process Synthesized videos

loc = ".../AllVideos/"  #location of videos
dst = ".../Frames/"	#destination of extracted frames
csv = ".../labelscsv/"	#label files(ground truth)


# -------------------------- Function Definitions ------------------------------

def getFrames(q):

    # function given to threads; thread goes into infinte loop and only exits with main
    # function is blocked at q.get() if q.get doesn't return anything
    # and so waits till something appears in the queue again.

    while True:
        vid = q.get()          # getting video info form queue

        vid_loc = vid[0]       # gets video location
        vid_fldr = vid[1]      # gets frame destination

        name = vid_loc[-14:-1] + vid_loc[-1]    # cutting out name from location path
        print "Processing:\t" + name

        container = av.open(vid_loc + ".mp4")

        for i, frame in enumerate(container.decode(video=0)):
            frame.to_image().save(vid_fldr + "%03d.jpg" % (i+1))

        print "Completed:\t" + name
        q.task_done()           # tells queue current job is done


def ensur_dir(file_path):

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):		#comment out to overwrite
        os.makedirs(directory)
    return


# ------------------------------- Main Code ------------------------------------

que = Queue()   # lists videos and keeps then in queue
vid_info = []   # list contains video source location and frame destination

# settings for synthesized videos
if syn:
    vidtype = "Synthesized"
    e = 12
else:
    vidtype = "Real"
    e = 13

loc = loc
dst = dst + vidtype +"/"

ensur_dir(dst)  # create main destination folder

# starting threads
for i in range(max_threads):
    if __name__ == '__main__':                                  # only main thread gets to call other threads
        t = threading.Thread(target=getFrames,args=(que,))      # assigning target to thread
        t.setDaemon(True)                                       # designating as Daemon Thread -> runs in bg
        t.start()                                               # starting thread

# getting video locations,setting up directories
for n in range (N[0],N[1]+1):                                            # video number

    data = numpy.genfromtxt(csv + "%02d_%02d.csv"%(syn+1,n),delimiter = ',')

    for eff in range(effects[0],effects[1]+1):                                         # effect type
        if syn == 1 and eff == 12:
            continue
        lvl = levels[0]
        while lvl<levels[1]+1:                                               # challenge level
            if eff ==0:
                vid_name = "%02d_%02d_00_00_00" % ((syn+1),n)
                lvl = 10
            else:
                vid_name = "%02d_%02d_01_%02d_%02d" % (syn+1,n,eff,lvl)
                lvl += 1

            vid_info = [(loc + vid_name),(dst + vid_name + "/")]    # queue is first in first out (FIFO)
            que.put(vid_info)                                       # (location, destination) put in queue

            ensur_dir(dst + vid_name + "/")                         # create destination sub-folder

que.join()           # blocks main until dameon threads have finished their job
print "Done !"
