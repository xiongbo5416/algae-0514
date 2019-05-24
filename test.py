# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:00:57 2019

@author: xiong
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import askopenfilename

#how many points used in alighment ?
NUM_POINTS=4
root = tkinter.Tk()
root.withdraw()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

fgbg = cv2.createBackgroundSubtractorMOG2()

#select lensfree image
name = "C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/2/"


for i in range(9):
    img = cv2.imread(name + str(1270+i) +".jpg")
    img = cv2.GaussianBlur(img,(19,19),0)
    fgmask = fgbg.apply(img)
 #   cv2.imshow('fgmask'+ str(i),fgmask)
    cv2.imshow('img',img)
    
    
for i in range(9):
    img = cv2.imread(name + str(1294+i) +".jpg")
    img = cv2.GaussianBlur(img,(15,15),0)
    fgmask = fgbg.apply(img)
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((4,4),np.uint8)
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
    fg_img = cv2.bitwise_and(img,img, mask= fgmask)
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('img',img)
    cv2.imshow('fg_img',fg_img)