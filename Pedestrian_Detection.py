import cv2
import time
import numpy as np

#Create your own classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#Initiate video capture for video file
cap = cv2.VideoCapture('people_walking.mp4')

while cap.isOpened():

    #time.sleep(0.5)   #this slows down the video....higher the number  more slow it is

    ret,frame = cap.read()
    
    # Resizing the video frame
    frame = cv2.resize(frame, None , fx = 0.5 , fy = 0.5, interpolation= cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mention the scalefactor(ex. 1.2) and minimum neighbors(ex. 3) 
    bodies = body_classifier.detectMultiScale(gray , 1.2, 3)

    #Draw rectangle when a object is identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y) , (x+w,y+h), (0,255,255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13:  #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

