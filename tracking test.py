# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:18:42 2019

@author: xiong
"""

import cv2
import dlib

PATH='C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/'
FOLDER_name= '2_lensfree/'

detector = dlib.simple_object_detector("detector.svm")


tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[1]

#if tracker_type == 'BOOSTING':
#    tracker = cv2.TrackerBoosting_create()
#if tracker_type == 'MIL':
#    tracker = cv2.TrackerMIL_create()
#if tracker_type == 'KCF':
#    tracker = cv2.TrackerKCF_create()
#if tracker_type == 'TLD':
#    tracker = cv2.TrackerTLD_create()
#if tracker_type == 'MEDIANFLOW':
#    tracker = cv2.TrackerMedianFlow_create()
#if tracker_type == 'GOTURN':
#    tracker = cv2.TrackerGOTURN_create()
#if tracker_type == 'MOSSE':
#    tracker = cv2.TrackerMOSSE_create()
#if tracker_type == "CSRT":
#    tracker = cv2.TrackerCSRT_create()

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()
trackerList = []

    
img = cv2.imread(PATH + FOLDER_name + str(1233) + '.jpg')
dets = detector(img)
print("Number of faces detected: {}".format(len(dets)))
#    for k, d in enumerate(dets):
#        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#            k, d.left(), d.top(), d.right(), d.bottom()))

i_object=0
for k, d in enumerate(dets):
    print("  <box top='{}' left ='{}' width ='{}' height ='{}'/>".format(
        d.top(), d.left(), d.right()-d.left(), d.bottom()-d.top()))
#        cv2.rectangle(gray_dif,(d.right(),d.bottom()),(d.left(),d.top()),(255),3)#use square to label the images 
    
        # Define an initial bounding box
    bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()) 
    # Initialize tracker with first frame and bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
    tracker = cv2.TrackerMOSSE_create()
    ok = tracker.init(img, bbox)
    print(bbox)
    trackerList.append(tracker)
    i_object=i_object+1
    
 
for i in range (10):
    img = cv2.imread(PATH + FOLDER_name + str(1233+i) + '.jpg')
    for j in range(i_object):
        ok, bbox = trackerList[j].update(img)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
    cv2.imshow(str(i),img)
    #use hog to find particls in the images
   