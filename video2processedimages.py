# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:51:13 2019

@author: xiong
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename
import dlib


# provide folder path and video name
PATH='C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/'
FOLDER_name= '2_lensfree/'
FOLDER_name_2='2_fl/'
NAME_lensfree='2_lensfr.h264'
NAME_scattering='2_fl.h264'
threshold_ex = 0 #set it to 0 when want to decide it 
#select a picture to check whether excitation light is on in the scattering/fl images
root = tkinter.Tk()
root.withdraw()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

#read onet frame to select high light area when threshold_ex=0
#
if threshold_ex==0:
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    img = cv2.imread(filename)
    bbox = cv2.selectROI(img, False)
    img_cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    threshold_ex = img_cut.sum()/10
    print(threshold_ex)

cap=cv2.VideoCapture(PATH +NAME_lensfree)
CONTRAST_EH=2 #contrast enhancement for fringes 
NUM_COMP=13

#creat a list to record the frame No. that is the first frame in each lasers toggle period
list_period = np.zeros(100,dtype=np.int32)
#record how many frames in the period
lensfr_period = np.zeros(100,dtype=np.int32)
fl_period = np.zeros(100,dtype=np.int32)

#load svm file for hog 
detector = dlib.simple_object_detector("detector.svm")

#purpose: alighment scattering/fl images to lensfree images
#point1 and points2 should be found mannual.point1s are from lensfree images
def alighment(width, height,gray):
    points1 = np.zeros((4, 2), dtype=np.float32)
    points2 = np.zeros((4, 2), dtype=np.float32)
    points1[0]=(1313,150)
    points1[1]=(1257,767)
    points1[2]=(558,710)
    points1[3]=(510,332)
    points2[0]=(1055,192)
    points2[1]=(965,670)
    points2[2]=(438,639)
    points2[3]=(399,364)
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # Use homography
    #height, width = gray_1.shape
    gray_reg = cv2.warpPerspective(gray, h, (width, height))
    return gray_reg


##convert lessfree video to images
#read first frame of video to get size of video
ret, frame = cap.read()
gray_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_float=gray_0.astype(float)
[ROL_NUM,COL_NUM]=gray_0.shape
shape_lensfr = gray_0.shape

#gray_0 shal be presented frame. gray_1 is bgd framewhich refers the previous frame
current_frame=0
period_num=0
RedOrNot=0
gray_1=gray_0*0  #set bgd to 0

while period_num<100:
    ret, frame = cap.read()
    
    
    #obtained blue channel of the present frame 
    blue = np.array(frame[:, :, 0])
    red = np.array(frame[:, :, 2])
    green = np.array(frame[:, :, 1])
    blue_float=blue.astype(float)
    
    #obtain gray images
    gray_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray_0=blue
    cv2.imshow('frame',red)
    
    
    
    gray_float=gray_0.astype(float)
 
    blue_profile=sum(blue_float.T)
    blue_profile=blue_profile/COL_NUM
    
    y_profile=sum(gray_float.T)
    y_profile=y_profile/COL_NUM

    #red laser is off
    if  max(blue_profile)>180 or min(y_profile)<10:
        RedOrNot=0
        gray_1=gray_0*0
        print(current_frame)
    
    # red laser just turn on. the first frame in this cycle
    elif gray_1.sum() == 0:
        RedOrNot=1
        gray_1=gray_0
        # save num of first frame in each lasers toggle period
        list_period[period_num]=current_frame+1000+1
        period_num=period_num+1
        
    # red laser has turned on. after second frame in this cycle
    else:   
        gray_dif = gray_0 - gray_1 +128#get difference between presented and previous frames
        gray_1=gray_0#renew background frame
        gray_dif=cv2.multiply(gray_dif-128+int(128/CONTRAST_EH),CONTRAST_EH)#enhacne contrast of gray images
        
        #save frames
        name= PATH + FOLDER_name +str(current_frame+1000) +'.jpg'
        
#        #use hog to find particls in the images
#        dets = detector(gray_dif)
#        print("Number of faces detected: {}".format(len(dets)))
#        #    for k, d in enumerate(dets):
#        #        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#        #            k, d.left(), d.top(), d.right(), d.bottom()))
#        for k, d in enumerate(dets):
#            print("  <box top='{}' left ='{}' width ='{}' height ='{}'/>".format(
#                d.top(), d.left(), d.right()-d.left(), d.bottom()-d.top()))
#            cv2.rectangle(gray_dif,(d.right(),d.bottom()),(d.left(),d.top()),(255),3)#use square to label the images
#        
#        #write note
#        font = cv2.FONT_HERSHEY_SIMPLEX
#        cv2.putText(gray_dif,'red laser on',(50,150), font, 4, (255), 4, cv2.LINE_AA)
        
        cv2.imwrite(name,gray_dif)
        #record num of frames in this period
        lensfr_period[period_num-1]=current_frame+1000
        #print(max(blue_profile))
        
    #press "Q" on the keyboard to quit
    if cv2. waitKey(1)& 0xFF == ord('q'):
        break
    
    current_frame += 1;
    
    
##convert scattering video to images
cap=cv2.VideoCapture(PATH +NAME_scattering)

#read first frame of video to get size of video
ret, frame = cap.read()
gray_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_float=gray_0.astype(float)
[ROL_NUM,COL_NUM]=gray_0.shape
gray_1=gray_0*0  #set bgd 

current_frame_fl=0
period_num=0
# excitation_st=0, excitation light is off, red light is on
# excitation_st=1, excitation light just on, red light just off
# excitation_st>2, excitation light must be on for whole frame
# excitation_st is the num of the frame. The red light has been off before the number of frame
excitation_st=0  


#for stable particles removel
fgbg = cv2.createBackgroundSubtractorMOG2()

while current_frame_fl<current_frame:
    ret, frame = cap.read()
    #cv2.imshow('frame_2',frame)
    gray_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray_highlight_sum= gray_0[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]].sum()
    print(str(current_frame_fl)+":"+ str(gray_highlight_sum))
    # if larger than threshold, the red light is on, excitation light is off
    if gray_highlight_sum > threshold_ex:
       excitation_st=0
       # when previous frame is performed, excitation light is on
       gray_1=gray_0*0
    #red light is on in the last frame,but off in this frame
    elif gray_1.sum()==0:
       excitation_st=1
       if current_frame_fl>2:
           period_num=period_num+1
       gray_1=gray_0
    #excitation light must be on for whole frame
    else:
       excitation_st = excitation_st+1
       if excitation_st>2:
           #save frames
           name= PATH + FOLDER_name_2 + str(list_period[period_num-1]+excitation_st-3+NUM_COMP) +'.jpg'
           #alighn scatteing/fl images to lensfree images
           gray_reg= alighment(shape_lensfr[1], shape_lensfr[0],gray_1)
           gray_reg= cv2.multiply(gray_reg,2)
           
           
           #gaussian filter to reduce noise
           gray_reg = cv2.GaussianBlur(gray_reg,(19,19),0)
           gray_reg= cv2.multiply(gray_reg,5)
           #apply background subtraction
           fgmask = fgbg.apply(gray_reg)
           #close precusure to remove noise black pixel inside the particles 
           kernel = np.ones((3,3),np.uint8)
           fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
           #close precusure to remove noise of white pixel outside the particles 
           fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
           #erode and dilate to remove the small particles
           kernel = np.ones((5,5),np.uint8)
           fgmask = cv2.erode(fgmask,kernel,iterations = 1)
           fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
           #apply mask to filter out stable particles
           fg_gray_reg = cv2.bitwise_and(gray_reg,gray_reg, mask= fgmask)

#           #use red square to label the fluorescent particles
#           lower = np.array([40])
#           upper = np.array([255])
#           shapeMask = cv2.inRange(fg_gray_reg, lower, upper)
#           cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#           edges=cnts[1]
#           for points in edges:
#                cv2.rectangle(fg_gray_reg,(points[0][0][0]-30,points[0][0][1]-30),(points[0][0][0]+30,points[0][0][1]+30),(255),3)
#           #use red square to label the fluorescent particles */
#
#            #write note
#           font = cv2.FONT_HERSHEY_SIMPLEX
#           cv2.putText(fg_gray_reg,'blue laser on',(50,150), font, 4, (255), 4, cv2.LINE_AA)
        
           cv2.imwrite(name,fg_gray_reg) 
           #record num of frames in this period
           fl_period[period_num-1]=list_period[period_num-1]+excitation_st-3+NUM_COMP
           #cv2.imwrite(name,cv2.multiply(gray_1,4)) 
           #print(gray_highlight_sum)
       gray_1=gray_0
       
    current_frame_fl=current_frame_fl+1
           

cap.release()
cv2.destroyAllWindows()