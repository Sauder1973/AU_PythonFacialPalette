# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:04:47 2021

@author: WesSa

This is not good - cannot adjust the frame rate.  Use the other process instead.


"""

# Program To Read video
# and Extract Frames
import cv2


Path = "F:"
TestVideo = "01-01-03-02-02-01-01.mp4"
fileName = Path +"/" + TestVideo


# Function to extract frames
def FrameCapture(path):
	
	# Path to video file
    vidObj = cv2.VideoCapture(path)
    vidObj.set(cv2.CAP_PROP_POS_MSEC, 15)

	# Used as counter variable
    count = 0

	# checks whether frames were extracted
    success = 1
    while success:

        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        #Show image:
        if success:
            cv2.imshow('image',image)
            cv2.waitKey(0)
            ##cv2.destroyAllWindows()



        # Saves the frames with frame-count
        #cv2.imwrite("frame%d.jpg" % count, image)
        count += 1

# Driver Code
#if __name__ == '__main__':

	# Calling the function
	#FrameCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4")

FrameCapture(fileName)
cv2.destroyAllWindows()
         