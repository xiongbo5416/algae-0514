# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:40:53 2019

@author: xiong
"""

import cv2
import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename

root = tkinter.Tk()
root.withdraw()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)


img = cv2.imread(filename)
print("  <image file='" + filename + "'>")

while(1):
    #select wanted area
    bbox = cv2.selectROI(img, False)
    print("    <box top='" + str(bbox[1]) + "' left ='" + str(bbox[0]) + "' width ='" + str(bbox[2]) + "' height ='" + str(bbox[3]) + "'/>" )
    img_cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    cv2.imshow('cut',img_cut)
    

cv2.destroyAllWindows()

#<box top='90' left='194' width='37' height='37'/>
# resize image
#dim = (WIDTH, HEIGHT)
#resized_img_cut = cv2.resize(img_cut, dim, interpolation = cv2.INTER_AREA)




#cv2.imwrite('C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/fringes.jpg', resized_img_cut)