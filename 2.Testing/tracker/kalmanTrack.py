
import numpy as np
from numpy.linalg import inv

class kalman:
	def __init__(self):
		self.dt = 1                                     # sampling rate
		self.acc = 0                                    # acceleration magnitude to start
		self.acc_noise = 2.3                            # uncertainty in acc
		self.noise_x = 0.1                              # noise along x-axis
		self.noise_y = 0.1                              # noise along y-axis

		self.Ez = np.matrix([[self.noise_x, 0],[0,self.noise_y]])
		self.Ex = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2,0],[0,(self.dt**4)/4,0,(self.dt**3)/2],[(self.dt**3)/2,0,(self.dt**2),0],[0,(self.dt**3)/2,0,self.dt**2]])
		self.Ex = self.Ex*(self.acc_noise**2)

		self.P = self.Ex

		self.A = np.matrix([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
		self.B = np.matrix([[(self.dt**2)/2],[(self.dt**2)/2],[self.dt],[self.dt]])
		self.C = np.matrix([[1,0,0,0],[0,1,0,0]])

		self.Q_estimate = None
		self.X_pre = np.zeros([1,10])
		self.Y_pre = np.zeros([1,10])

		self.flag = True                                  # flag = 0 for first frame
		self.curr_kidx = []                     	      # kalaman index is used to label each bounding box for assignment to tracks

	def track(self,bboxes,kidx):
		try:
			if kidx == [] or bboxes == []:
				return []

			nboxes = len(bboxes)
			X = [bboxes[i][0] for i in range(nboxes)]
			Y = [bboxes[i][1] for i in range(nboxes)]

			if not self.flag:
				keepID = sorted([ID for ID in range(len(self.curr_kidx)) if self.curr_kidx[ID] in kidx])            # move these cols to the left side
				addID = sorted([ID for ID in range(len(kidx)) if kidx[ID] not in self.curr_kidx])                   # add these cols to the right
				self.Q_estimate = self.Q_estimate[:,keepID]
				self.Q_estimate = np.hstack((self.Q_estimate,np.zeros([4,len(addID)])))


			t1 = np.vstack([X,Y])							# t1 is 2xN sizes array
			t2 = np.vstack([t1,np.zeros([2,nboxes])])       # t2 is 4xN size array
			Q_measured = np.asmatrix(t1)                    # it is a matrix of 2xN
			Q_measured = Q_measured.T                       # it is a matrix of Nx2
			Q = np.asmatrix(t2)                             # Q is a matrix of 4xN

			self.curr_kidx = kidx							# keepign kalman index of boxes being currently tracked

			if self.flag:                                	# for 1st frame predicted value and measured values are same
				self.Q_estimate = Q
				self.flag = False

			P_estimate = self.P

			for i in range(nboxes):
				self.Q_estimate[:,i] = self.A * self.Q_estimate[:,i] + self.B * self.acc

			# predict next covariance
			self.P = self.A * self.P * self.A.T + self.Ex

			# kalman gain
			K = self.P * self.C.T * inv(self.C * self.P * self.C.T + self.Ez)

			for i in range(nboxes) :
				self.Q_estimate[:,i] = self.Q_estimate[:,i] + K * (Q_measured[i,:].T - self.C * self.Q_estimate[:,i])


			I = np.identity(4)
			I = np.asmatrix(I)

			self.P = (I - K * self.C) * self.P

			X_pred = np.array(self.Q_estimate[0,:])
			Y_pred = np.array(self.Q_estimate[1,:])

			bboxed_pred = np.array([(X_pred[0][i],Y_pred[0][i]) for i in range(nboxes)],dtype=np.uint)

			return bboxed_pred

		except:
			return []

	def reset(self):
		self.dt = 1
		self.acc = 0
		self.acc_noise = 2.3
		self.noise_x = 0.1
		self.noise_y = 0.1

		self.Ez = np.matrix([[self.noise_x, 0],[0,self.noise_y]])
		self.Ex = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2,0],[0,(self.dt**4)/4,0,(self.dt**3)/2],[(self.dt**3)/2,0,(self.dt**2),0],[0,(self.dt**3)/2,0,self.dt**2]])
		self.Ex = self.Ex*(self.acc_noise**2)

		self.P = self.Ex

		self.A = np.matrix([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
		self.B = np.matrix([[(self.dt**2)/2],[(self.dt**2)/2],[self.dt],[self.dt]])
		self.C = np.matrix([[1,0,0,0],[0,1,0,0]])

		self.Q_estimate = None
		self.X_pre = np.zeros([1,10])
		self.Y_pre = np.zeros([1,10])

		self.flag = True
		self.curr_kidx = []
