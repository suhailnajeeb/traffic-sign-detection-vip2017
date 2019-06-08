
from kalmanTrack import *
from cnn.cnnLib import scaleBoxes
import numpy as np
import cv2
import scipy as sp
from scipy import spatial
from collections import Counter

# k-d tree, returns nearest neighbour
def kdtreeNN(refpoints,intpoints):
    mytree = sp.spatial.cKDTree(refpoints)
    dist,indexes = mytree.query(intpoints)
    return indexes

# k-d tree, returns points within a circle of radius = max_distance
def kdtree(refpoints,intpoints,max_distance):
    tree = sp.spatial.cKDTree(refpoints)
    indexes = tree.query_ball_point(intpoints,max_distance)
    return indexes

# main tracker class
class hybridTracker:
    def __init__(self):
        self.track_mode = 'LK'                                  # default tracking filter = Lucas-Kanade Tracker
        self.track_len = 150                                    # max history of a certain tracked point kept for 150 frames
        self.detect_interval = 3                                # get new features every 3 frames, and predict in between
        self.tracks = []                                        # tracked reference points
        self.trackedBoxes = {}                                  # tracked bounding boxes


        self.num_bboxes = 0                                     # number of boxes being tracked
        self.boxlife = 18                                       # starting life of each bounding box
        self.fidx = 0                                           # current frame number
        self.challengeType = []                                 # keeps track of challengeType of each frame as classifed at frame generator

        self.kalman_idx = 0                                     # index to keep track of boxes when using kalman filter
        self.useKalmanFor = ['Shadow','DirtyLens','Bright']     # default tracking system switched to kalman for these challenge types
        self.kalmanTracker = kalman()                           # creating kalman tracker object
        self.kalman_pts = []                                    # kalman predictions kept here as buffer; used as prediction, when track_mode = 'Kalman'

        # {0:'NoCh',1:'Blurry',2:'Dark',3:'Bright',4:'Noisy',5:'Shadow',6:'Snow',7:'Haze',8:'DirtyLens'}

        # parameters for LK - tracker
        self.lk_params = dict(  winSize  = (30, 30),
                                maxLevel = 7,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


        # parameters for picking the best points to track (optical flow)
        self.feature_params = dict( maxCorners = 2000,
                                    qualityLevel = 0.2,
                                    minDistance = 1,
                                    blockSize = 7 )

    # resets tracker for processing new video
    def reset(self):
        self.track_mode = 'LK'
        self.tracks = []
        self.trackedBoxes = {}
        self.num_bboxes = 0
        self.fidx = 1
        self.challengeType = []
        self.kalman_idx = 0
        self.kalmanTracker.reset()
        self.kalman_pts = []

    # updates track mode depending on type of challenge/environmental conditions
    def updateTrackMode(self):

        count = Counter(self.challengeType)
        max_chtype = max(count,key=count.get)       # determing the challenge type detected the most in preceeding frames

        if max_chtype in self.useKalmanFor:         # using appropriate tracking mode
            self.track_mode = 'Kalman'
        else:
            self.track_mode = 'LK'

    # cleans up dead boxes
    def cleanUpBoxes(self):

        # if CNN feedback states bbox not ROI ...
        nonROI_idx = [num for num in range(self.num_bboxes) if self.trackedBoxes[num]['ROI'] == False]

        # ... box's life is decreased -> to consecutive ROI=False feedbacks will kill box faster
        for boxID in nonROI_idx:
            self.trackedBoxes[boxID]['life'] -= self.boxlife/2

        # if life is less than <threshold>, box is considered dead (box allowed to persist even after life = 0 to prevent repeated detections of that box)
        deadbox_idx = [num for num in range(self.num_bboxes) if self.trackedBoxes[num]['life'] <-self.boxlife/1.75]

        for boxID in deadbox_idx:
            del self.trackedBoxes[boxID]

        self.num_bboxes -= len(deadbox_idx)

        # re-labelling boxes that are alive
        i=0
        validBoxes = {}
        for key in self.trackedBoxes:
            validBoxes[i] = self.trackedBoxes[key]
            validBoxes[i]['updated'] = False
            i+=1

        self.trackedBoxes = validBoxes


    # takes feedback from CNN, marks boxes
    def feedback(self,classOut):

        classOut = np.array(classOut)                                       # getting class labels from sign classifier
        nonROI_idx = np.where(np.logical_or(classOut>14,classOut<1))[0]     # determing box IDs classed as non-sign
        ROI_idx = np.where(np.logical_and(classOut<15,classOut>0))[0]       # determining box IDS classed as signs

        for boxID in nonROI_idx:
            try:
                self.trackedBoxes[boxID]['ROI'] = False                     # disabling ROI flag for boxes deemed as non ROI
            except KeyError:
                continue

        for boxID in ROI_idx:
            try:
                self.trackedBoxes[boxID]['ROI'] = True                      # enabling/reenabling ROI flag for boxes deemed as ROI
            except KeyError:
                continue


    # update states,locations,sizes etc. of bounding boxes
    def update(self,bboxes):

        # remove dead boxes
        self.cleanUpBoxes()

        # no curenntly tracked boxes, all new boxes
        if self.num_bboxes == 0:
            for box in bboxes:
                self.trackedBoxes[self.num_bboxes] = {}
                self.trackedBoxes[self.num_bboxes]['cood'] = (box[0],box[1])            # current co-ordinates (center of box)
                self.trackedBoxes[self.num_bboxes]['predcood'] = None                   # predicted co-ordinates (updated by tracker)
                self.trackedBoxes[self.num_bboxes]['size'] = (box[2],box[3])            # box size
                self.trackedBoxes[self.num_bboxes]['ROI'] = True                        # ROI flag
                self.trackedBoxes[self.num_bboxes]['birth'] = self.fidx                 # frame in which detected
                self.trackedBoxes[self.num_bboxes]['life'] = self.boxlife               # life of box
                self.trackedBoxes[self.num_bboxes]['class'] = None                      # class box is deemed as (updated by sign classifier)
                self.trackedBoxes[self.num_bboxes]['updated'] = False                   # flag to determine if box has been updated by tracker/bbox detector output
                self.trackedBoxes[self.num_bboxes]['vel'] = (0,0)                       # velocity of box
                self.trackedBoxes[self.num_bboxes]['acc'] = (0,0)                       # acceleration of box
                self.trackedBoxes[self.num_bboxes]['kidx'] = self.kalman_idx            # unique index to be used by kalman tracker
                self.kalman_idx = (self.kalman_idx + 1)%1000                            # kalamn index loops after 1000
                self.num_bboxes += 1

        elif self.num_bboxes != 0:
            if len(bboxes>0):                                                                               # FRCNN forwards boxes
                new_pts = bboxes[:,:2]                                                                      # seperating the center co-cordinates
                new_pts_sizes = bboxes[:,2:]                                                                # and the sizes of the bbox
                curr_pts = np.array([self.trackedBoxes[num]['cood'] for num in range(self.num_bboxes)])     # Retrieving co-ordinates of current tracked boxes

                samebox_idx = kdtree(curr_pts,new_pts,100)              # finding nearest neighbours

                for i,boxID in enumerate(samebox_idx):

                    # no matches, all are new points
                    if not boxID:
                        self.trackedBoxes[self.num_bboxes] = {}
                        self.trackedBoxes[self.num_bboxes]['cood'] = (bboxes[i][0],bboxes[i][1])
                        self.trackedBoxes[self.num_bboxes]['predcood'] = None
                        self.trackedBoxes[self.num_bboxes]['size'] = (bboxes[i][2],bboxes[i][3])
                        self.trackedBoxes[self.num_bboxes]['ROI'] = True
                        self.trackedBoxes[self.num_bboxes]['birth'] = self.fidx
                        self.trackedBoxes[self.num_bboxes]['life'] = self.boxlife
                        self.trackedBoxes[self.num_bboxes]['class'] = None
                        self.trackedBoxes[self.num_bboxes]['updated'] = False
                        self.trackedBoxes[self.num_bboxes]['vel'] = (0,0)
                        self.trackedBoxes[self.num_bboxes]['acc'] = (0,0)
                        self.trackedBoxes[self.num_bboxes]['kidx'] = self.kalman_idx
                        self.kalman_idx = (self.kalman_idx + 1)%1000
                        self.num_bboxes += 1
                    else:
                        # if more than one match, get nearest neighbour from those that match
                        if len(boxID) > 1:
                            boxID = kdtreeNN(curr_pts,new_pts[i])
                        else:
                            boxID = boxID[0]

                        # now compare prediction with FRCNN co-ordinates
                        xF,yF = new_pts[i]
                        xsizeF,ysizeF = new_pts_sizes[i]
                        xprev,yprev = self.trackedBoxes[boxID]['cood']
                        xvel,yvel = self.trackedBoxes[boxID]['vel']
                        xacc,yacc = self.trackedBoxes[boxID]['acc']
                        age = self.fidx - self.trackedBoxes[boxID]['birth']

                        # check if prediction exists
                        if self.trackedBoxes[boxID]['predcood'] is not None:
                            xP,yP = self.trackedBoxes[boxID]['predcood']

                            # if diffence between prediction and FRCNN is greater than 30px, use prediction (for LK tracker)
                            if ((xF-xP)**2 + (yF-yP)**2) > 900 and self.track_mode == 'LK':
                                self.trackedBoxes[boxID]['cood'] = (int(xP),int(yP))
                                self.trackedBoxes[boxID]['size'] = (xsizeF,ysizeF)
                                self.trackedBoxes[boxID]['updated'] = True
                                self.trackedBoxes[boxID]['vel'] = (xP-xprev,yP-yprev)
                                self.trackedBoxes[boxID]['acc'] = ((xP-xprev)-xvel,(yP-yprev)-yvel)
                                self.trackedBoxes[boxID]['predcood'] = None
                                if self.trackedBoxes[boxID]['ROI']:
                                    self.trackedBoxes[boxID]['life'] = self.boxlife

                            # if FRCNN output exists for box, use regardless of prediction (for Kalman tracker)
                            elif self.track_mode == 'Kalman':
                                self.trackedBoxes[boxID]['cood'] = (int(xF),int(yF))
                                self.trackedBoxes[boxID]['size'] = (xsizeF,ysizeF)
                                self.trackedBoxes[boxID]['updated'] = True
                                self.trackedBoxes[boxID]['vel'] = (xP-xprev,yP-yprev)
                                self.trackedBoxes[boxID]['acc'] = ((xP-xprev)-xvel,(yP-yprev)-yvel)
                                self.trackedBoxes[boxID]['predcood'] = None
                                if self.trackedBoxes[boxID]['ROI']:
                                    self.trackedBoxes[boxID]['life'] = self.boxlife

                            # else use FRCNN output
                            else:
                                self.trackedBoxes[boxID]['cood'] = (xF,yF)
                                self.trackedBoxes[boxID]['size'] = (xsizeF,ysizeF)
                                self.trackedBoxes[boxID]['updated'] = True
                                self.trackedBoxes[boxID]['vel'] = (0,0)
                                self.trackedBoxes[boxID]['acc'] = (0,0)
                                self.trackedBoxes[boxID]['predcood'] = None
                                if self.trackedBoxes[boxID]['ROI']:
                                    self.trackedBoxes[boxID]['life'] = self.boxlife

                        # prediction doesn't exist, so use FRCNN output regardless
                        else:
                            self.trackedBoxes[boxID]['cood'] = (xF,yF)
                            self.trackedBoxes[boxID]['size'] = (xsizeF,ysizeF)
                            self.trackedBoxes[boxID]['updated'] = True
                            self.trackedBoxes[boxID]['vel'] = (0,0)
                            self.trackedBoxes[boxID]['acc'] = (0,0)
                            if self.trackedBoxes[boxID]['ROI']:
                                self.trackedBoxes[boxID]['life'] = self.boxlife

            # if current frame is 2 or higher
            if self.fidx>1:

                # check whether anyboxes left to be updated (updated=False => FRCNN didnt have output for that box)
                nobox_idx = [num for num in range(self.num_bboxes) if self.trackedBoxes[num]['updated'] is False]

                if nobox_idx:
                    for boxID in nobox_idx:

                        xprev,yprev = self.trackedBoxes[boxID]['cood']
                        xvel,yvel = self.trackedBoxes[boxID]['vel']
                        xacc,yacc = self.trackedBoxes[boxID]['acc']
                        xP2,yP2 = (xprev + xvel + 0.5*xacc , yprev + yvel + 0.5*yacc)
                        age = self.fidx - self.trackedBoxes[boxID]['birth']

                        if self.trackedBoxes[boxID]['predcood'] is not None:
                            xP,yP = self.trackedBoxes[boxID]['predcood']

                            self.trackedBoxes[boxID]['cood'] = (int(xP),int(yP))
                            self.trackedBoxes[num]['updated'] = True
                            self.trackedBoxes[boxID]['vel'] = (xP-xprev,yP-yprev)
                            self.trackedBoxes[boxID]['acc'] = ((xP-xprev)-xvel,(yP-yprev)-yvel)
                            self.trackedBoxes[boxID]['predcood'] = None
                        else:
                            if age>1:
                                self.trackedBoxes[boxID]['cood'] = (int(xP2),int(yP2))
                                self.trackedBoxes[num]['updated'] = True
                                self.trackedBoxes[boxID]['vel'] = (xP2-xprev,yP2-yprev)
                                self.trackedBoxes[boxID]['acc'] = ((xP2-xprev)-xvel,(yP2-yprev)-yvel)
                            else:
                                self.trackedBoxes[num]['life'] = -1

            # Natural Decay in Box Life
            for boxID in self.trackedBoxes:
                self.trackedBoxes[boxID]['life'] -= 1.5


    # updates predicted co-ordinates using LK prediction
    def LKPred(self,P):
        # reference pts
        ref_pts = P[:,:2]

        # if reference pts exists, get co-ordinates of bboxes
        if len(ref_pts>0):
            bbox_pts = np.array([self.trackedBoxes[num]['cood'] for num in range(self.num_bboxes)])

            # index of bbox-nearest neigbouring reference pts
            pidx = kdtreeNN(ref_pts,bbox_pts)

            # neigbours found, now determine which points best to use for prediction
            for boxID,(idx,point) in enumerate(zip(pidx,bbox_pts)):

                vel = P[idx,2:4]

                (x,y) = self.trackedBoxes[boxID]['cood']

                xP = x + vel[0]
                yP = y + vel[1]

                self.trackedBoxes[boxID]['predcood'] = (xP,yP)

        # if ref pts not found, forgo prediction
        else:
            return

    # updates predicted co-ordinates using kalman prediction
    def KalmanPred(self):
        if self.kalman_pts == []:
            return

        for boxID,pred in enumerate(self.kalman_pts):
            self.trackedBoxes[boxID]['predcood'] = (pred[0],pred[1])

    # main tracking function
    def track(self,LKTrackQ,cnnOutQ,bboxQ,cnnQ,trackerOutQ):
        while True:

            self.fidx +=1

            if self.fidx>301:
                ## print "\n-------------------\n Resetting Tracker \n-------------------\n"
                self.reset()

            ch_type,frame_gray = LKTrackQ.get()            # grayscale image and challenge type
            self.challengeType.append(ch_type)
            LKTrackQ.task_done()

            # getting output from CNN for feedback
            if self.fidx >1:
                classOut = cnnOutQ.get()
                self.feedback(classOut)
                cnnOutQ.task_done()


            # LK prediction being prepped
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray                     # needs two frames, current and previous

                # getting track points of current frame
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                # calculating trackpoints for next frame
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)

                # reverse calculating trackpoints for current frame using predicted points
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)

                # difference between reverse calc and actual
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                D = np.int16((p1-p0).reshape(-1,2))

                # the prediction is good if difference less than 2px
                good = d < 2

                P = np.int16(p0.reshape(-1,2))
                P = np.hstack((P,D,good.reshape(-1,1)))

                # take only good points; P contains X,Y,dx,dy
                # X,Y are trackpoints of current frame, and dx,dy are the changes
                # in their position according to the prediction
                P = P[P[:,4]==1,0:4]

                # Kalman prediction being prepped
                if self.num_bboxes > 0:
                    bbox_pts = np.array([self.trackedBoxes[num]['cood'] for num in range(self.num_bboxes)])         # obtaining co-ordinates (centers) of boxes
                    kidx = [self.trackedBoxes[num]['kidx'] for num in range(self.num_bboxes)]                       # obtaining kalman index of boxes (needed for assignment to tracks)
                    self.kalman_pts = self.kalmanTracker.track(bbox_pts,kidx)                                       # obtaining kalman filter predictions and storing

                # predict bbox locations if num of bboxes > 0
                if self.num_bboxes > 0:

                    # using LK tracking filter
                    if self.track_mode == 'LK':
                        self.LKPred(P)


                    # usig kalman filter to predict
                    elif self.track_mode == 'Kalman':
                        self.KalmanPred()


                new_tracks =[]
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    # if predictions arent good, discard point
                    if not good_flag:
                        continue

                    # else keep points
                    tr.append((x, y))

                    # if more than the specified number of points taken, delete oldest to make room
                    if len(tr) > self.track_len:
                        del tr[0]

                    new_tracks.append(tr)

                self.tracks = new_tracks


            # get new good features to track every <self.detect_interval> frames
            if (self.fidx-1) % self.detect_interval == 0:

                # mask layer
                mask = np.zeros_like(frame_gray)

                # clear mask
                mask[:] = 255

                # fill mask at locations of ref pts...
                ## for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                ##     cv2.circle(mask, (x, y), 5, 0, -1)

                # ...  and bboxes
                for x, y in np.int32([self.trackedBoxes[num]['cood'] for num in range(self.num_bboxes)]):
                    cv2.circle(mask, (x, y), 10, 0, -1)

                # gets strong corners on an image near regions marked in the mask, used as tracking points
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)

                # if good feature points obtained, append them to tracks
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):           # p reshaped into columns, each row correspond to x,y pair
                        self.tracks.append([(x, y)])

            # saving frame for act as prev frame for next one
            self.prev_gray = frame_gray

            # select tracking system based on challenge type
            if (self.fidx) % 7 == 0:
                prev_mode = self.track_mode
                self.updateTrackMode()
                if prev_mode != self.track_mode:
                    print "\n--------------------------------------"
                    print "Tracking System changed to : %s"%self.track_mode
                    print "--------------------------------------\n"

            # getting bboxes from FRCNN
            bboxes = np.array(bboxQ.get())
            bboxQ.task_done()

            if len(bboxes)>0:
                # converting to x,y,w,h
                bboxes[:,0] = (bboxes[:,0]+bboxes[:,2])//2
                bboxes[:,1] = (bboxes[:,1]+bboxes[:,3])//2
                bboxes[:,2] = 2*(bboxes[:,2]-bboxes[:,0])
                bboxes[:,3] = 2*(bboxes[:,3]-bboxes[:,1])

            # update states of boxes
            self.update(bboxes)

            # putting updated tracked boxes on the Queue for CNN
            bboxesOut = np.array([self.trackedBoxes[num]['cood'] for num in range(self.num_bboxes)])
            bboxesSize = np.array([self.trackedBoxes[num]['size'] for num in range(self.num_bboxes)])
            nonROI_idx = [num for num in range(self.num_bboxes) if self.trackedBoxes[num]['life'] <=0]
            bboxesSize[nonROI_idx] = (0,0)
            bboxesOut = np.hstack((bboxesOut,bboxesSize))

            if len(bboxesOut)>0:
                # converting back to x1,y1,x2,y2
                bboxesOut[:,0] = bboxesOut[:,0] - bboxesOut[:,2]//2
                bboxesOut[:,2] = bboxesOut[:,0] + bboxesOut[:,2]
                bboxesOut[:,1] = bboxesOut[:,1] - bboxesOut[:,3]//2
                bboxesOut[:,3] = bboxesOut[:,1] + bboxesOut[:,3]

            # forwarding bounding box to classifier
            cnnQ.put(bboxesOut)

            # bounding box also made accessible for visualization
            trackerOutQ.put(bboxesOut)
