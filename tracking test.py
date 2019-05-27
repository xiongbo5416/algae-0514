# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:18:42 2019

@author: xiong
"""

import cv2
import dlib
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

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

particle_List = []
detect_p_list = []
count_particles=0
lensfree_st=[1209,1233,1257,1281,1306,1330,1354,1378,1402]

class particle:
    MAX_F= 6
    def __init__(self, bbox,img):
      self.bbox = bbox
      self.img = img
      self.position_x = [0, 0]
      self.position_y = [0, 0]
      self.frame= [0, 0]
      self.speed = [0, 0]
      self.appear = 5
    
    def tracker_create(self):
        self.tracker = cv2.TrackerCSRT_create()
    
    def get_save_position(self,frame_num):
        self.position_x.append(int (self.bbox[0] + self.bbox[2]/2))
        self.position_y.append(int (self.bbox[1] + self.bbox[3]/2))
        self.frame.append(frame_num)       
        if len(self.frame)> particle.MAX_F:
            del self.position_x[0]
            del self.position_y[0]
            del self.frame[0]
            
    def report_missing(self):
         self.appear = self.appear-1
         if self.appear < 0:
             self.appear = 0
             self.speed[0]=0
             self.speed[1]=0
    
    def get_speed(self):
        if len(self.frame) > particle.MAX_F-1:
            self.speed[0]= (self.position_x[particle.MAX_F-1] - self.position_x[2]) / (self.frame[particle.MAX_F-1]-self.frame[2])
            self.speed[1]= (self.position_y[particle.MAX_F-1] - self.position_y[2]) / (self.frame[particle.MAX_F-1]-self.frame[2])  
        else:
            self.speed[0]=0
            self.speed[1]=0
        return self.speed
    
    def positon_predict(self, frame):
        if self.speed[1] > 0 :
            move_y = (frame - self.frame[particle.MAX_F-1])*self.speed[1]
            move_x = (frame - self.frame[particle.MAX_F-1])*self.speed[0]
            pos_p=(self.position_x[particle.MAX_F-1]+move_x, self.position_y[particle.MAX_F-1]+move_y)
        else:
            pos_p=(-100,-100)
            
        return  pos_p

        #print(self.position_y)
        
#   def tracker_init(self):
#      ok = self.tracker.init(self.img, self.bbox)
#      print(ok)
      
#img = cv2.imread(PATH + FOLDER_name + str(1209) + '.jpg')
#
#print("Number of faces detected: {}".format(len(dets)))
##    for k, d in enumerate(dets):
##        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
##            k, d.left(), d.top(), d.right(), d.bottom()))

font = cv2.FONT_HERSHEY_SIMPLEX
        
i_object=0
#print(particle_List[2].bbox)
jpg_i=0

for f in glob.glob(os.path.join(PATH + FOLDER_name, "*.jpg")):
    img = cv2.imread(f)
    frame_num= f[-8:-4]
    frame_num = int(frame_num)
    print(frame_num) 
    #current object number
    i_object_cru=i_object
    
    if frame_num in lensfree_st:
        dets = detector(img)
        for k, d in enumerate(dets):
            #print("  <box top='{}' left ='{}' width ='{}' height ='{}'/>".format(
            #    d.top(), d.left(), d.right()-d.left(), d.bottom()-d.top()))
        #        cv2.rectangle(gray_dif,(d.right(),d.bottom()),(d.left(),d.top()),(255),3)#use square to label the images 
            
                # Define an initial bounding box
            bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
            position_cru=(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            # Initialize tracker with first frame and bounding box
            
            #find the min distrance between position and predict position
            min_distance=1000
            object_found=-1
            for j in range(i_object_cru):
                postison_pred= particle_List[j].positon_predict(frame_num)
#                p1 = (int(postison_pred[0]), int(postison_pred[1]))
#                p2 = (int(postison_pred[0] + 5), int(postison_pred[1] + 5))
#                cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
#                cv2.putText(img,str(j),p1, font, 1, (255,255,255), 1, cv2.LINE_AA)
                #print(postison_pred)
                distance=pow(postison_pred[0]-position_cru[0],2) + pow(postison_pred[1]-position_cru[1],2)
                distance=pow(distance,0.5)
                if distance<min_distance:
                    min_distance=distance
                    object_found=j
                    
            if min_distance >60:
                particle_a= particle(bbox, img)
                particle_a.tracker_create() 
                ok = particle_a.tracker.init(img, bbox)
                particle_List.append(particle_a)
                #draw square when find it using hog
#                p1 = (int(bbox[0]), int(bbox[1]))
#                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#                cv2.rectangle(img, p1, p2, (255,255,0), 3, 2)
                
                
                i_object=i_object+1
            else:    
                particle_List[object_found].tracker_create()
                particle_List[object_found].tracker.init(img, bbox)
                print(str(object_found) + 'found')
                #print(object_found)
                #print(postison_pred)
            
            print(min_distance)
    
    for j in range(i_object):
        if particle_List[j].appear > 0:
            ok, bbox = particle_List[j].tracker.update(img)
            if not ok:
                print('frame:'+ str(frame_num) + '  object:'+str(j))
                particle_List[j].report_missing()
                
            elif (bbox[1]>1100):
                particle_List[j].report_missing()
                
            else:
                particle_List[j].bbox=bbox
                if particle_List[j].get_speed()[1]>2:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
#                    cv2.putText(img,str(j),p1, font, 1, (255,255,255), 1, cv2.LINE_AA)
                    #counting
                    if j not in detect_p_list:
                        detect_p_list.append(j)
                        count_particles=count_particles+1
                else:
                    particle_List[j].report_missing()

                    
                    
                particle_List[j].get_save_position(frame_num)
                
    jpg_i = jpg_i +1     
    cv2.putText(img,'Counting: ' + str(count_particles),(200,200), font, 3, (255,255,255), 3, cv2.LINE_AA)
    cv2.imwrite(f,img) 
#    if frame_num==1209 or frame_num==1233 or frame_num==1266:
#        cv2.imshow(str(frame_num),img)
#        #imgplot = plt.imshow(img)
#        #plt.show()


for j in range(i_object):
    print(str(j) + ' : ' + str(particle_List[j].get_speed()))

#print (particle_List[10].position_y)
print (particle_List[10].position_y[2:-1])
print (particle_List[10].frame)
print (particle_List[24].position_x)

   