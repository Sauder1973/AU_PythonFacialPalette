# -*- coding: utf-8 -*-
"""

Standard Procedures

1 - ROI Generator
"""



import cv2 
from feat import Detector
from numpy import asarray
import numpy as np
from autocrop import Cropper
from PIL import Image
from skimage.feature import hog
from skimage import data, color, exposure
import pandas as pd
from matplotlib import pyplot as plt
import os
from imutils import build_montages
from imutils import face_utils


# import the necessary packages
from imutils import face_utils
import matplotlib.pyplot as plt
import imutils
import dlib
import cv2
import torch


           # imageIn = image_Cropped.copy()
           # LandmarkData =  np_faceONE
           # FacialLandmark = "mouth"


def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.figure(figsize=(12, 12))
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()
    
    
    

# ROI - Region of Interest Generator -------------------------------------------------------------------------------
def ImageROI (imageIn, LandmarkData, FacialLandmark, Verbose = True):

    AllowedFeatures = ["mouth","right_eyebrow","left_eyebrow", "right_eye","left_eye","nose", "jaw" ]



    featureAllowed = FacialLandmark in AllowedFeatures

    if featureAllowed:
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS[FacialLandmark]
        
        rStart = int(rStart)
        rEnd   = int(rEnd)
    
    
    # # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([LandmarkData[rStart:rEnd]]))
        x = x
        y = y 
        h = h
        roi = imageIn[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    		
    # # # show the particular face part
    
        if Verbose == False:
            currFeat = "ROI: " + str(FacialLandmark)
            plt_imshow(currFeat, roi)
     
    return roi






def ExtendedImageROI (imageIn, LandmarkData, FacialLandmark, Verbose = True):
    testImageCropped = imageIn
    np_faceONE = LandmarkData
    
    AllowedFeatures = ["mouth", "right_eye","left_eye" ]

    print("In Procedure - at facial landmark:" + FacialLandmark)

    featureAllowed = FacialLandmark in AllowedFeatures
    #proceed = True
    roi = 0
    if featureAllowed:

        if FacialLandmark == "right_eye":
            print("Right Eye in Procedure")
            LandmarkReferences = [17,18,19,20,21, 27,40,41]
            proceed = True
        if FacialLandmark == "left_eye":
            print("Left Eye in Procedure")
            LandmarkReferences = [27,22,23,24,25,26,46,47]
            proceed = True
        if FacialLandmark == "mouth":
            print("At Mouth in Procedure")
           #LandmarkReferences = [3,13,8]  #All Lower Jaw
            LandmarkReferences = [6,33,10]  #All Lower Jaw
            proceed = True
    else:
        proceed = False
        roi = 0
        print("ERROR - Feature NOT ALLOWED")
            
    if proceed == True:
        # # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([np_faceONE[LandmarkReferences]]))
        print("X:" + str(x), " Y:" + str(y) + " w:" + str(w) +" h:" + str(h))
        widthAdjust = 0
        heightAdjust = 0
        roi = testImageCropped[y:y + h + heightAdjust, x:x + w + widthAdjust]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    # # # show the particular face part

        if Verbose == False:
            print("Trying to Show Image")
            currFeat = "ROI: " + str(FacialLandmark)
            plt_imshow(currFeat, roi)
    
    print("Leaving Procedure:::: ROI = "  )
    return roi
            
    



def PaletteMontage(result_Images, Height = 128, Width = 180, Rows = 14, Columns = 6, Verbose = True, Path = "", FileFamily = ""):
    ## construct the montages for the images
    
    # montages = build_montages(result_Images, (256, 330), (7, 3))
    montages = build_montages(result_Images, (128, 180), (14, 6))
    i = 1
    
    ## loop over the montages and display and or save each of them
    for montage in montages:
        i = i + 1
        if Path != "":
            fileName = FileFamily + "_" + i
            SaveFileName = Path + "/" + fileName
            cv2.imwrite(SaveFileName,montage)

        if Verbose == False:         
            cv2.imshow("Montage - Cluster: %d" % (i), montage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            
def k_medoids(similarity_matrix, k):
    
    # Step 1: Select initial medoids
    num = len(similarity_matrix)
    row_sums = torch.sum(similarity_matrix, dim=1)
    normalized_sim = similarity_matrix.T / row_sums
    normalized_sim = normalized_sim.T
    priority_scores = -torch.sum(normalized_sim, dim=0)
    values, indices = priority_scores.topk(k)
    
    tmp = -similarity_matrix[:, indices]
    tmp_values, tmp_indices = tmp.topk(1, dim=1)
    min_distance = -torch.sum(tmp_values)
    cluster_assignment = tmp_indices.resize_(num)
    print(min_distance)
    
    # Step 2: Update medoids
    for i in range(k):
        sub_indices = (cluster_assignment == i).nonzero()
        sub_num = len(sub_indices)
        sub_indices = sub_indices.resize_(sub_num)
        sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
        sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
        sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
        sub_medoid_index = torch.argmin(sub_row_sums)
        # update the cluster medoid index
        indices[i] = sub_indices[sub_medoid_index]
        print(i)
        
    # Step 3: Assign objects to medoids
    tmp = -similarity_matrix[:, indices]
    tmp_values, tmp_indices = tmp.topk(1, dim=1)
    total_distance = -torch.sum(tmp_values)
    cluster_assignment = tmp_indices.resize_(num)
    print(total_distance)
        
    while (total_distance < min_distance):
        min_distance = total_distance
        # Step 2: Update medoids
        for i in range(k):
            sub_indices = (cluster_assignment == i).nonzero()
            sub_num = len(sub_indices)
            sub_indices = sub_indices.resize_(sub_num)
            sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
            sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
            sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
            sub_medoid_index = torch.argmin(sub_row_sums)
            # update the cluster medoid index
            indices[i] = sub_indices[sub_medoid_index]
            print(total_distance)

        # Step 3: Assign objects to medoids
        tmp = -similarity_matrix[:, indices]
        tmp_values, tmp_indices = tmp.topk(1, dim=1)
        total_distance = -torch.sum(tmp_values)
        cluster_assignment = tmp_indices.resize_(num)
        print(total_distance)
        
    return indices

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.figure(figsize=(12, 12))
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster ):
    
    import pytz
    import datetime    

    montages = build_montages(dataFacePart,(frameWidth, frameHeight), (montageCols, montageRows))
    i = 0
    FileFamily = FacePart
    
    print("Path equals: %s" %(Path))
    ## loop over the montages and display and or save each of them
    for montage in montages:
        i = i + 1
        #if Path != "":
        if Path != None:
            
            #Add DateStamp to Filename:
                
            my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
            
            dateVar = str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:])
            
            print("SAVING TO DISK")
            fileName = FileFamily + "_" + str(i) + dateVar
            SaveFileName = Path + "/" + fileName + ".jpg"
            print("SAVING: " + SaveFileName)
            cv2.imwrite(SaveFileName,montage)
        else: 
            
            cv2.imshow("Montage - Face Part: %s Cluster: %d  Frames: %d" % (FacePart,Cluster, i), montage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



def Montage_Cluster ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, ClusterLabel ):
    
    import pytz
    import datetime 
    import cv2

    montages = build_montages(dataFacePart,(frameWidth, frameHeight), (montageCols, montageRows))
    i = 0
    FileFamily = FacePart
                    
    my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
    dateVar = str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:])
    
    print("Path equals: %s" %(Path))
    ## loop over the montages and display and or save each of them
    for montage in montages:
        if Path != None:
            
            print("SAVING TO DISK")
            fileName = FileFamily + "_" + dateVar
            SaveFileName = Path + "/" + fileName + "_CLUSTER_" + str(ClusterLabel) +".jpg"
            print("SAVING: " + SaveFileName)
            cv2.imwrite(SaveFileName,montage)
        else: 
            i = i + 1
            cv2.imshow("Montage - Face Part: %s Cluster: %d  Frames: %d" % (FacePart,ClusterLabel, i), montage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

