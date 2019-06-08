import numpy as np
import os

label_dir = './labels/'
save_dir = './labels2_5/'
zoom_factor = 2.5

# makes sure x-coordinates are within frame after displace/zoom
def roundx(x):
	return 0 if x<0 else 1627 if x>1627 else x


# makes sure y-coordinates are within frame after displace/zoom
def roundy(y):
	return 0 if y<0 else 1235 if y>1235 else y


# zooms out ROI
def zoomROI(llx,lly,urx,ury,zoom_factor=2.5):

	# width,height of ROI
	x = abs(urx-llx)
	y = abs(ury-lly)

	# calculating amount to move each corner of ROI
	dx = np.round((zoom_factor*x - x)/2.0)
	dy = np.round((zoom_factor*y - y)/2.0)

	return roundx(llx-dx),roundy(lly-dy),roundx(urx+dx),roundy(ury+dy)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for t in [1,2]:
	for n in range(1,50):
		try:
			data = np.genfromtxt(label_dir + '%02d_%02d.txt'%(t,n),delimiter = '_')		# loading Ground truth labels
			llx = data[1:,2]
			lly = data[1:,3]
			urx = data[1:,8]
			ury = data[1:,9]

			for i in range(len(llx)):
				lx,ly,ux,uy = zoomROI(llx[i],lly[i],urx[i],ury[i],zoom_factor=zoom_factor)	# only modifiying opposite conrner co-ordinates
				data[1:,2][i] = lx															# since only these are used in the rest of the system
				data[1:,3][i] = ly
				data[1:,8][i] = ux
				data[1:,9][i] = uy

			data = np.int16(data[1:,:])
			np.savetxt(save_dir + '%02d_%02d.csv'%(t,n),data,delimiter=',',header="frame,sign,llx,lly,lrx,lry,ulx,uly,urx,ury")
		except:
			np.savetxt(save_dir + '%02d_%02d.csv'%(t,n),data,delimiter=',',header="frame,sign,llx,lly,lrx,lry,ulx,uly,urx,ury")
			continue
