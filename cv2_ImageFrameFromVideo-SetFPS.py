# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:22:13 2021

@author: WesSa


# This process works very good!!!!
# Use this process as required.

"""

import cv2 
from feat import Detector
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)



Path = "F:"
TestVideo = "01-01-03-02-02-01-01.mp4"
fileName = Path +"/" + TestVideo

vidcap = cv2.VideoCapture(fileName) 
def getFrame(sec): # taken from : https://www.quora.com/How-do-I-decrease-the-frames-per-second-in-OpenCV-python
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
    hasFrames,image = vidcap.read() 
    if hasFrames:
        #Show image:
        cv2.imshow('image',image)
        cv2.waitKey(0)
        #image_prediction = detector.detect_image(image)
        # Show results
        #image_prediction
        #image_prediction.plot_detections();
        image_prediction = detector.detect_image(success)
        

        
#        cv2.imwrite("frame "+str(sec)+" sec.jpg", image)     # save frame as JPG file 
    return image_prediction


sec = 0 

fps = 30
frameRate = fps/360  ##it will capture image in each 0.5 second 
success = getFrame(sec) 
while success: 
    sec = sec + frameRate 
    sec = round(sec, 2) 
    success = getFrame(sec)
    #getFrame.hasFrames
    

    # Show results
    #image_prediction
    success.plot_detections();
    
cv2.destroyAllWindows()
