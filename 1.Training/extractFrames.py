# This code gets frames from each of the videos of the data set
# Change the location and destination of videos and frames in the settings section
# Set max number of threads and number of video to process as well
# Recommended number of threads = 15 for core i7
# To process Synthesized videos, set syn = 1

# Requires Videos to be in one folder, i.e all Real videos in a "Real" Folder
# and all Synthesized videos in a "Synthesized" Folder

import threading
from Queue import Queue
import os
import av
import numpy

# -------------------------- Function Definitions ------------------------------

# function to get frames
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

# ensures directory exists, if not creates one
def ensur_dir(file_path):

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):		#comment out to overwrite
        os.makedirs(directory)
    return

#--------------------------------- Settings ------------------------------------

max_threads = 7 # number of maximum active threads at a given time

N = [1,49]              # number of videos to process , 1-49
effects = [0,12]        # effect types 0-12 -> [0,0] implies no challenge
levels =  [1,5]         # effect levels 1-5
syn = 1                 # set to 1 to include Synthesized videos

# directory where videos are kept
loc = "/media/shahruk/Terra 2.0C/VIP Cup 2017 Data/Videos/"

# directory where frames will be saved
dst = "../Frames/"


# ------------------------------- Main Code ------------------------------------

que = Queue()   # lists videos and keeps then in queue
vid_info = []   # list contains video source location and frame destination


ensur_dir(dst)  # create main destination folder

# starting threads
for i in range(max_threads):
    if __name__ == '__main__':                                  # only main thread gets to call other threads
        t = threading.Thread(target=getFrames,args=(que,))      # assigning target to thread
        t.setDaemon(True)                                       # designating as Daemon Thread
        t.start()                                               # starting thread

# getting video locations,setting up directories
for t in range(1,2+syn):

    for n in range (N[0],N[1]+1):                                            # video number

        for eff in range(effects[0],effects[1]+1):                                         # effect type
            if t == 2 and eff == 12:
                continue

            lvl = levels[0]

            while lvl<levels[1]+1:                                               # challenge level
                if eff ==0:
                    vid_name = "%02d_%02d_00_00_00" % (t,n)
                    lvl = 10
                else:
                    vid_name = "%02d_%02d_01_%02d_%02d" % (t,n,eff,lvl)
                    lvl += 1

                ensur_dir(dst + vid_name + "/")                         # create destination sub-folder
                vid_info = [(loc + vid_name),(dst + vid_name + "/")]    # queue is first in first out (FIFO)
                que.put(vid_info)                                       # (location, destination) put in queue


que.join()           # blocks main until dameon threads have finished their job
print "Done !"
