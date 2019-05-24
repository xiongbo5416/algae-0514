# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:31:43 2019

@author: xiong
"""
import os
import sys
import glob
import cv2

faces_folder = "C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/2"

img_array = []
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    img = cv2.imread(f)
    img = img[0:1000,0:1300,:]
    img_array.append(img)
    height, width, layers = img.shape
    size = (width,height)
    print(f)
    
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


 
