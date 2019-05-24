import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal
import xlwt
import cv2
import pylab
from scipy import ndimage



#get the index of line with max blue light
#img=mpimg.imread('0ppm_bgd0.png')
#red= np.array(img[:,:,0])
#blue=np.array(img[:,:,2])

#plt.plot(sum(blue.T))
#get index of peak line
#index_b_peak=np.argmax(sum(blue.T))


#read measured image png
#img=mpimg.imread('0ppm0.png')
#red= np.array(img[:,:,0])
#blue=np.array(img[:,:,2])

#read measured image jpg
#img=mpimg.imread('C:/Users/xiong/OneDrive - McMaster University/algae_project/0407/scattering_high resolution.jpg')
im = cv2.imread('C:/Users/xiong/OneDrive - McMaster University/algae_project/0407/scattering_high resolution.jpg')
#blue= np.array(im[:,:,0])
#red=np.array(im[:,:,2])
#green=np.array(im[:,:,1])
RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
maxValue = 255
adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
blockSize = 5 #odd number like 3,5,7,9,11
C = -3 # constant to be subtracted
im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C) 
labelarray, particle_count = ndimage.measurements.label(im_thresholded)
print(particle_count)
pylab.figure(1)
pylab.imshow(RGB_im)
pylab.show()