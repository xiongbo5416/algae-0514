# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:22:49 2019

@author: xiong
"""

import os
import sys
import glob
import cv2
import dlib

faces_folder = "C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/2"


detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

print("Showing detections on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    #print(format(f))
    print(f)
    #img = dlib.load_rgb_image(f)
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
#    for k, d in enumerate(dets):
#        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#            k, d.left(), d.top(), d.right(), d.bottom()))
    for k, d in enumerate(dets):
        print("  <box top='{}' left ='{}' width ='{}' height ='{}'/>".format(
            d.top(), d.left(), d.right()-d.left(), d.bottom()-d.top()))
        cv2.rectangle(img,(d.right(),d.bottom()),(d.left(),d.top()),(255,255,255),3)

    cv2.imshow('img',img)
    cv2.imwrite(f,img) 