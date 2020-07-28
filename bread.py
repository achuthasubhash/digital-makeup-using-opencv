# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:00:26 2020

@author: DELL
"""

import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy

jewel_img = cv2.imread("jewelery.png")
frame = cv2.imread('priyanka.jpeg')
frame = cv2.resize(frame,(432, 576)) #resize frame

# Returns a list of face landmarks present on frame
face_landmarks_list = face_recognition.face_landmarks(frame)
# For demo images only one person is present in image 
face_landmarks = face_landmarks_list[0]  #0 so 1 persom

shape_chin = face_landmarks['chin'] #print list of points of chin
# x,y cordinates on frame where jewelery will be added
x = shape_chin[0][0]  #start from 4th point  so 3 ,0 give x coord,1 give y coord
y = shape_chin[0][1]
img_width = abs ( shape_chin[0][0] - shape_chin[17][0])  
img_height = int(x* img_width) #based on jewllery height and width did else disort happens
jewel_img = cv2.resize(jewel_img, (img_width,img_height), interpolation=cv2.INTER_AREA)  #area downsize area
jewel_gray = cv2.cvtColor(jewel_img, cv2.COLOR_BGR2GRAY)
# All pixels greater than 230 will be converted to white and others will be converted to black
thresh, jewel_mask = cv2.threshold(jewel_gray, 230, 255, cv2.THRESH_BINARY)
# Convert to black the background of jewelry image bec add black it be adding
jewel_img[jewel_mask == 255] = 0  #white to black
# Crop out jewelry area from original frame
jewel_area = frame[y:y+img_height, x:x+img_width]
# bitwise_and will convert all black regions in any image to black in resulting image
masked_jewel_area = cv2.bitwise_and(jewel_area, jewel_area, mask=jewel_mask)
# add both images so that the black region in any image will result in another image non black regions being rendered over that area
final_jewel = cv2.add(masked_jewel_area, jewel_img)
# replace original frame  jewel area with newly created jewel_area
frame[y:y+img_height, x:x+img_width] = final_jewel
plt.show(frame)