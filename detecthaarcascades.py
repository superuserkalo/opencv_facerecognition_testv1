import cv2
import numpy
import os
import time

face_cascade = cv2.CascadeClassifier('hfsb.xml')

vid = cv2.VideoCapture(0) 

if not vid.isOpened():
        raise IOError("Cannot open webcam")  
  
while True: 
    ret, frame = vid.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    
    biggest_face = (0,0,0,0)

    for (x, y, w, h) in faces:
        if w > biggest_face[2]:
            biggest_face = (x,y,w,h)

    if biggest_face != (0,0,0,0):
        x = biggest_face[0]
        y = biggest_face[1]
        w = biggest_face[2]
        h = biggest_face[3]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 

    cv2.imshow('Camera', frame) 

    c = cv2.waitKey(1) 
    if c == 27:
        break


vid.release() 

cv2.destroyAllWindows() 

