# -*- coding: utf-8 -*-
"""
# STAGE TWO:
# Facial PALETTE Code:
#   Wes Sauder
#    2200442
#   Athabasca University

#  February 27, 2022

# This is the primary code for building facial parsing process.
#   Steps to follow:
#       Retrieve Data From Facial Parser
#           - Pkl Files for HOG
#           - Raw Images
#           - AU/Index etc
#       Frames to HOG (OPTIONAL)
#       
#       PCA ANALYSIS - Dimensional Reduction
#       Clustering
#       SSIM (or alternative) - Image Similiarity Scoring
#           Determine quality In Cluster (minimum distance) and Between Cluster (maximum distance)
#           Remove Outlier from Cluster
#       Stage Data with Cluster Folders and File to Cluster File for PyTorch Dataloader in next process (Facial Reenactment)


@author: WesSa

"""



#Load CSV from Facial Parser:

# 1 - Action Units


# Housekeeping:
import cv2 as cv
import pandas as pd
from feat import Detector as featDetect
import numpy as np
import pyFeatHOG as HOG

from matplotlib import pyplot as plt

import os
import pickle

# Constants


# Setup HOG Modeling
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"

detector = featDetect(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)


# Setup Image and HOG paths from Stage One - Facial Parser
LeftEyeImagesPath = "F:\Project_FacialReenactmentOutputs\Actor_12\FacesOut_2022_02_26_HR_2049_OutFinal\Images_RAW_CROPPED\LeftEye_2022_02_26_HR_2055"
RightEyeImagesPath = "F:\Project_FacialReenactmentOutputs\Actor_12\FacesOut_2022_02_26_HR_2049_OutFinal\Images_RAW_CROPPED\RightEye_2022_02_26_HR_2056"
MouthImagesPath = "F:\Project_FacialReenactmentOutputs\Actor_12\FacesOut_2022_02_26_HR_2049_OutFinal\Images_RAW_CROPPED\Mouth_2022_02_26_HR_2056"

ImagePaths = [LeftEyeImagesPath,RightEyeImagesPath,MouthImagesPath]
FacialFeature = ["LeftEye", "RightEye", "Mouth"]

#Get List of Files:
filesForAnalysis = []
fileDetails = []
    
for root, dirs, files in os.walk(ImagePaths[1]):
    for file in files:
        if file.endswith(".jpg"):
            fileToAnalyze = os.path.join(root,file)
            df_tmp = {'FullPathToAnalyze': fileToAnalyze,
                        'File': file,
                        'FacialFeature':FacialFeature[1]}            
            fileDetails.append(df_tmp)
            
            #print(fileToAnalyze)
            filesForAnalysis.append(fileToAnalyze)


images_FacialPart = pd.DataFrame(fileDetails)
images_FacialPartFilesWithPath = fileDetails[["FullPathToAnalyze"]]


# Open PKL File - HOG numpy Array from Stage One: Facial Parser

pickleHOG_Root = 'F://Project_FacialReenactmentOutputs//Actor_12//FacesOut_2022_02_26_HR_2049_OutFinal//HOG_DATA//'

with open(pickleHOG_Root+'LeftEye_HOG_Data.pkl', 'rb') as f:
    result_LeftEye_HOG_fd_FLAT = pickle.load(f)

# -----------------------------------------------------------------------------------------------------------------------------------------------





# -----------
# -----------------------------------------------------------------------------------------------------------------------------

# Similarity Matrix


import torch
import time
useTorch = True

num = len(result_LeftEye_HOG_fd_FLAT)
similarity_matrix = np.zeros((num, num))

HOG_Matrix_torch = np.concatenate(result_LeftEye_HOG_fd_FLAT)
HOG_Matrix_torch = torch.from_numpy(HOG_Matrix_torch)

for i in range(0, num):
    t0 = time.time()
    print("Currently at i: ",i, "At Time : ",t0)
    for j in range(i+1, num):
        if useTorch == True:
            diff =  torch.sub(HOG_Matrix_torch[i], 
                              HOG_Matrix_torch[j])
            #diff = diffTorch
        else:
            diffReg = HOG_Matrix_torch[i] - HOG_Matrix_torch[j]
            # diff = diffReg
        
       #Code To Validate the Torch and the Numpy Approaches
        print(type(diff))
        print("Torch Difference = ",torch.sum(diff))
        # print(type(diffReg))
        # print("Regular Subtraction Difference = ",np.sum(diffReg))
        
       # diffCheck = diffTorch - diffReg
       # print("Type of Result ",type(diffCheck))
       # print("Result of the difference of each data type: ",diffCheck)
       # print("Torch Sum of the Difference of each Type = ",torch.sum(diffCheck))
        
      #  if (i%10 == 0 & j%10 == 0):
      #      print("i:",i," j:",j)
        
        dist_tmp = np.linalg.norm(diff)
        similarity_matrix[i][j] = dist_tmp
        similarity_matrix[j][i] = dist_tmp
    print (time.time() - t0, " seconds wall time")


type(similarity_matrix)

torch_SimMatrix = torch.from_numpy(similarity_matrix)


#convert to list
list_SimMatrix = torch_SimMatrix.tolist()
#flatten the lists
flattened_list = [y for x in list_SimMatrix for y in x]

import numpy as np
import random
from matplotlib import pyplot as plt

data_Flat =flattened_list

# fixed bin size
bins = np.arange(0,100) # fixed bin size

plt.xlim([min(data_Flat)-5, max(data_Flat)+5])

plt.hist(data_Flat, bins=bins, alpha=0.5)
plt.title('Similarity Matrix Histogram')
plt.xlabel('Distance')
plt.ylabel('Count')

plt.show()


indices = k_medoids(torch_SimMatrix, k=3)
indices





 
# END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END
# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# If PKL file does not exist - rebuild HOG using the following:

currFile = images_FacialPartFilesWithPath["FullPathToAnalyze"].loc[1]

for currFile in images_FacialPartFilesWithPath:
        print("Current File: " + currFile)
        
        #Open Current Image
        
        currImage = cv.imread(currFile)
        
        #Determine the number of channels:
        currImage.ndim
        
        #Convert to HOG - Extract Histogram of Gradients - HOG        
        orientations = 8
        pixels_per_cell = (8,8)
        cells_per_block = (2,2)
        multiChannel = True
        dpi = 60
        
       
        PyFeat_fd,PyFeat_hog_image = detector.extract_hog(currImage, orientation=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)#, multichannel = multiChannel)


        plt.imshow(PyFeat_hog_image, interpolation='nearest')
        plt.show()