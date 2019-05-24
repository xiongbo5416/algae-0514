# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:47:14 2019

@author: xiong
"""

# import the necessary packages
import numpy as np
import cv2
import tkinter
from tkinter.filedialog import askopenfilename

root = tkinter.Tk()
root.withdraw()
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
img = cv2.imread(filename)
    
lower = np.array([25, 25, 25])
upper = np.array([255, 255, 255])
shapeMask = cv2.inRange(img, lower, upper)
cv2.imshow('mask',shapeMask)

cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edges=cnts[1]

for points in edges:
    cv2.rectangle(img,(points[0][0][0]-30,points[0][0][1]-30),(points[0][0][0]+30,points[0][0][1]+30),(0,0,255),5)

cv2.imwrite(filename,img) 
cv2.imshow('img',img)
