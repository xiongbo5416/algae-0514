# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:51:13 2019

@author: xiong
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

PATH='C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/'
FILENAME='2_fl'
cap=cv2.VideoCapture(PATH + FILENAME + '.h264')

current_frame=0;
i_frame=0;
while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    name= PATH + FILENAME +str(current_frame) +'.jpg'
    print(current_frame)
    gray_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(name,cv2.multiply(gray_0,1)) 
    #cv2.imwrite(name,frame)
    
    if cv2. waitKey(1)& 0xFF == ord('q'):
        break
    current_frame += 1;
    
    

cap.release()
cv2.destroyAllWindows()