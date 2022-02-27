# -*- coding: utf-8 -*-
"""
# STAGE ONE:
# Facial Parsing Code:
#   Wes Sauder
#    2200442
#   Athabasca University

#  February 21, 2022

# This is the primary code for building facial parsing process.
#   Steps to follow:
#       Video to Frames
#       Frames to HOG
#       Clustering
#       File Saves to Disk
#       Output Summaries to Disk



# This process works very good!!!!
# Use this process as required.

"""

import cv2 
from feat import Detector
from numpy import asarray
import numpy as np
from autocrop import Cropper
from PaletteProc import ImageROI  
from PaletteProc import ExtendedImageROI  
from PaletteProc import k_medoids
from PaletteProc import plt_imshow
from PaletteProc import image_resize
from PaletteProc import Montage_FacePart
from PIL import Image
from skimage.feature import hog
from skimage import data, color, exposure
import pandas as pd
from matplotlib import pyplot as plt

from DataLoading import FaceLandmarksDataset
from DataLoading import FacialDataset

from SaveImages import Save_FileToClusterDirctory
from SaveImages import Save_FilesToSingleDirctory
from SaveImages import Save_FilesToSingleDir_NoCluster

import seaborn as sns
import os
from pathlib import Path
import pickle


import DateTime
import datetime
import pytz

#import imutils
from imutils import build_montages
from imutils import face_utils

import dlib


from fast_pytorch_kmeans import KMeans
import torch
from torch.utils.data import Dataset



# HOUSEKEEPING: -------------------------------------------------------------------------------------------------
    
    
fps = 60  #(Higher the Value - The Lower the Frame Rate)

frameRateSuperHigh = 15

frameRateHigh = 30
frameRateMed  = 60
frameRateLow  = 120


frameRateSelected = 15  #Last Run

contrastLevel = 1.0  #Use this to adjust the image for Regular Image
contrastLevel_HOG = 2.0  #Use this to adjust the image for HOG Imaging


face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)



# initialize unique color for each facial landmark region
colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
    (168, 100, 168), (158, 163, 32),
    (163, 38, 32), (180, 42, 220),
    (100, 68, 109)]



#Individual Frames:

# Path = "F:"
# TestVideo = "01-01-03-02-02-01-01.mp4" 
# TestVideo = "02-01-05-01-01-01-12 - Angry Jane.mp4" 
# fileName = Path +"/" + TestVideo


# # # TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_16_10_37_Pro.mp4"
# # # TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_16_16_45_Pro.mp4"
# #TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_17_17_58_Pro.mp4"
# #TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_17_34_18_Pro.mp4"

# #fileName = TestVideo


# Code to Flatten List of Lists
def flattenListOfLists(t):
    return [item for sublist in t for item in sublist]



def dictToCSV(path, fileString, dictData):
    import csv

    a_file = open(path+fileString, "w")
    writer = csv.writer(a_file)

    for key, value in dictData.items():
        writer.writerow([key, value])

    a_file.close()


# List of Videos to Analyze - From Directory ------------------------------------------------------------------------------

RyersonAV_Path = "F:/RyersonAV/RyersonActorFiles/ActorVideoOnly"
CurrentActor = "Actor_12"
RyersonAV_OutPath = "F:/ImageFrameFromVideo_TrialResults/Actor_12_Results"

RyersonAV_Eyes_OutPath = "F:/ImageFrameFromVideo_TrialResults/Actor_12_Results/Eyes"
RyersonAV_Mouth_OutPath = "F:/ImageFrameFromVideo_TrialResults/Actor_12_Results/Mouth"

path = RyersonAV_Path +"/" + CurrentActor
outPlots = False   # Use this to create the visuals showing each frame with AU and Landmark renderings.

filesForAnalysis = []

fileDetails = []


for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".mp4"):
            fileToAnalyze = os.path.join(root,file)
            df_tmp = {'FullPathToAnalyze': fileToAnalyze,
                        'File': file,
                        'Modality':  file[0:2],
                        'VocalChnnl':file[3:5],
                        'Emotion':   file[6:8],
                        'Intensity': file[9:11],
                        'Statement': file[12:14],
                        'Repetition':file[15:17],
                        'Actor':     file[18:20]}            
            fileDetails.append(df_tmp)
            
            
            print(fileToAnalyze)
              
            filesForAnalysis.append(fileToAnalyze)


fileDetails = pd.DataFrame(fileDetails)
fileDetails[["File"]]


FilesToProcess = fileDetails.loc[fileDetails.Repetition == '01', "FullPathToAnalyze"]

#fileName = filesForAnalysis[93]



# BEGIN FACIAL PARSING -----------------------------------------------------------------------------------------------
            
# Setup storage for outputs - 

result_AU           = {} # SET ONE: Action Units: AU
result_HOG          = {} # SET TWO: Histogram of Gradients (HOG)
result_FaceDim      = {} # SET THREE: Facial Dimensions
result_FaceLandmark = {} # SET FOUR: Face X, Y Landmarks
result_Emotion      = {} # SET FIVE: Emotions
result_Image        = {} # SET SIX: Cropped Image

result_Images       = []

result_RightBrow = [] 
result_LeftBrow  = []  
result_RightEye  = []  
result_LeftEye   = [] 
result_Mouth     = []

result_RightBrow_HOG = []
result_LeftBrow_HOG  = []
result_RightEye_HOG  = [] 
result_LeftEye_HOG   = []
result_Mouth_HOG     = []

result_RightBrow_HOG_fd = []
result_LeftBrow_HOG_fd  = []
result_RightEye_HOG_fd  = [] 
result_LeftEye_HOG_fd   = []
result_Mouth_HOG_fd     = []




result_INDEX = {}

j = 0


fileName = FilesToProcess.get(key = 0)

for fileName in FilesToProcess:
    print(fileName)
    print("File Number: " + str(j))

    # In the Wild - ME at the Computer
    #fileName = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211115_22_58_41_Pro.mp4"
    
    vidcap = cv2.VideoCapture(fileName) 
    
    
    frameRate = frameRateSelected/360  ##it will capture image in each 0.5 second 
    sec = 0 
    
    success = True
    
    # Load the cascade
    harrcascadePath = "C:/Users/WesSa/Python37_VENV/pyfeat37\Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
    #face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(harrcascadePath)
    
    facesMade = 0
    i = 0
    
    while success: 
        
        print("Frame Number: " + str(i))
        currFile = fileName
        facesMade = facesMade + 1
        sec = sec + frameRate 
        sec = round(sec, 2) 
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
        hasFrames,OrigImage = vidcap.read() 
        
        #print(hasFrames)
    
        if hasFrames: 
    
            #Preview Image:  - Colour is off since it is BGR NOT RGB
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(OrigImage, cv2.COLOR_BGR2RGB))
            # plt.show()
        
            # ADJUST CONTRAST, RESIZE AND CONVERT TO GRAYSCALE
            # plt.imshow(gray, interpolation='nearest')
            
            image = OrigImage
            image = cv2.addWeighted(OrigImage, contrastLevel, np.zeros(OrigImage.shape, OrigImage.dtype), 0, 0)
            
            # plt.imshow(cv2.cvtColor(OrigImage, cv2.COLOR_BGR2RGB))
            # plt.show()
            
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()
            
            
            image = image_resize(image,height= 1000)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
            # plt.axis("off")
            # plt.imshow(image)
            # plt.show()
    
        
            # Use Cropper to Detect and Crop Face
            cropper = Cropper(face_percent = 50)
            image_Cropped = cropper.crop(image)
     
    
            croppedImageFound = type(image_Cropped)
            
            if croppedImageFound is not type(None):
                i = i + 1
                # plt.axis("off")
                # plt.imshow(image_Cropped)
                # plt.show()
         
                   
                detected_faces = detector.detect_faces(image_Cropped)
                detected_landmarks = detector.detect_landmarks(image_Cropped, detected_faces)
                detected_landmarks = np.round(detected_landmarks, 0)
                emotions = detector.detect_emotions(image_Cropped, detected_faces, detected_landmarks)
             
                #Convert detected landmarks to numpy array
                detected_landmarks_np = np.asarray(detected_landmarks) 
                #detected_landmarks_np.shape
                detected_landmarks_np = detected_landmarks_np.transpose(2,0,1).reshape(68,-1)
    
                
                # Extract Histogram of Gradients - HOG        
                orientations = 8
                pixels_per_cell = (8,8)
                cells_per_block = (2,2)
                multiChannel = True
                dpi = 60
                
                
                
                #NEW NEW NEW - MODIFIED CODE FOR HOG AND LANDMARK DETECTION TO BE CONTROLLED BY A HIGH CONTRAST IMAGE.
                # Added line below with contrastLevel_HOG and new iamge_Cropped_HOG to be used in the extract_hog function
                
                image_Cropped_HOG = cv2.addWeighted(image_Cropped, contrastLevel_HOG, np.zeros(image_Cropped.shape, image_Cropped.dtype), 0, 0)
                
                PyFeat_fd,PyFeat_hog_image = detector.extract_hog(image_Cropped_HOG, orientation=orientations, 
                                          pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
    
                # Action Unit Derivation --- WORKS WORKS WORKS WORKS !!!!                
                processedFramesPyFeat = detector.process_frame(image_Cropped)
                
                
                # Pull the resulting dataframe apart to capture primary ingredients.
                Face_AU = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('AU')]
                Face_Coord_X = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('x_')]
                Face_Coord_Y = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('y_')]
                Face_Coords_XY = pd.concat([Face_Coord_X, Face_Coord_Y], axis=1)
                Face_DimDetails = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('Face')]
                Face_Emotion = processedFramesPyFeat[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]
        
        # Build Matrix for Landmark Data --------------------------------------------
                 
                tempX = Face_Coord_X.transpose()
                tempY = Face_Coord_Y.transpose()
          
                tempX.reset_index(drop=True, inplace=True)
                tempY.reset_index(drop=True, inplace=True)
                
                
                faceONE = [tempX, tempY ]
                faceONE = pd.concat(faceONE, axis=1, keys=None)
                faceONE.columns = ["X","Y"]
                faceONE["Y"] = faceONE["Y"]
                faceONE["X"] = faceONE["X"]
        
                
        # Plot Results from Emotional Analysis --------------------------------------------
            
                Face_Emotion_Long = Face_Emotion
                Face_Emotion_Long['FaceID'] = facesMade
                
                Face_Emotion_Long = Face_Emotion_Long.set_index('FaceID').T
                Face_Emotion_Long['Emotion'] = Face_Emotion_Long.index
                Face_Emotion_Long.index = (str('00000000' + str(facesMade))[-7:] + "_" + Face_Emotion_Long['Emotion'] )
                Face_Emotion_Long['FaceID'] = facesMade
                Face_Emotion_Long = Face_Emotion_Long.set_axis(['Score','Emotion','Face_ID'], axis=1, inplace=False)
                
                Plot_emotion = Face_Emotion_Long['Emotion'].tolist()
                Plot_score = Face_Emotion_Long['Score'].tolist()
                
                #plt.barh(Plot_emotion,Plot_score)
                
        # Plot Results from Action Units --------------------------------------------
            
                Face_AU_Long = Face_AU
                Face_AU_Long['FaceID'] = facesMade
                
                Face_AU_Long = Face_AU_Long.set_index('FaceID').T
                Face_AU_Long['ActionUnit'] = Face_AU_Long.index
                Face_AU_Long.index = (str('00000000' + str(facesMade))[-7:] + "_" + Face_AU_Long['ActionUnit'] )
                Face_AU_Long['FaceID'] = facesMade
                Face_AU_Long = Face_AU_Long.set_axis(['Score','ActionUnit','Face_ID'], axis=1, inplace=False)
                
                Plot_ActionUnit = Face_AU_Long['ActionUnit'].tolist()
                AU_Plot_score = Face_AU_Long['Score'].tolist()
                
     
                np_faceONE = faceONE.to_numpy()
                np_faceONE = np_faceONE.astype(int) 
    
                
         # Process each Facial Feature --------------------------------------------
                BasicAllowedFeatures = ["mouth","right_eyebrow","left_eyebrow", "right_eye","left_eye","nose", "jaw" ]
                ExtraAllowedFeatures = ["mouth", "right_eye","left_eye"]
                
                
                # Use this to simplify the Facial Parser:  IE Use ExtraAllowedFeatures for Eyes and Mouth
                AllowedFeatures = ExtraAllowedFeatures
                feat = ExtraAllowedFeatures[0]
                
                for feat in AllowedFeatures:
                    currFeat = "ROI: " + str(feat)
                
                    ROI_Returned  = ExtendedImageROI (image_Cropped, np_faceONE, feat, Verbose = True)
                    
                    #Plot ROI:
                    # plt.axis("off")
                    # plt.imshow(ROI_Returned)
                    # plt.show()
                    
                # Standardize the size of the HOG image.
    
                    if feat == "mouth":
                        Hog_Length = 250 
                        Hog_Width = 125
                    if feat == "right_eyebrow":
                        Hog_Length = 250 
                        Hog_Width = 125
                    if feat == "left_eyebrow":
                        Hog_Length = 250 
                        Hog_Width = 125
                    if feat == "right_eye":
                        Hog_Length = 250 
                        Hog_Width = 125
                        
                    if feat == "left_eye":
                        Hog_Length = 250 
                        Hog_Width = 125
                    
                    ROI_Returned = cv2.resize(ROI_Returned, dsize=(Hog_Length, Hog_Width), interpolation=cv2.INTER_CUBIC)
                    
                    

                    
                        
                    PyFeat_fd_ROI,PyFeat_ROI_hog_image = detector.extract_hog(ROI_Returned, orientation=orientations, 
                                              pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
                    
                    print("AT FEATURE: " + feat)
                    
                    if feat == "mouth":
                        #print("MAIN AT mouth: " + feat)
                        result_Mouth.append(ROI_Returned)
                        result_Mouth_HOG.append(PyFeat_ROI_hog_image)
                        result_Mouth_HOG_fd.append(PyFeat_fd_ROI)
                    if feat == "right_eyebrow":
                        #print("MAIN AT right_eyebrow: " + feat)
                        result_RightBrow.append(ROI_Returned)
                        result_RightBrow_HOG.append(PyFeat_ROI_hog_image)
                        result_RightBrow_HOG_fd.append(PyFeat_fd_ROI)
                    if feat == "left_eyebrow":
                        #print("MAIN AT left_eyebrow: " + feat)
                        result_LeftBrow.append(ROI_Returned)
                        result_LeftBrow_HOG.append(PyFeat_ROI_hog_image)
                        result_LeftBrow_HOG_fd.append(PyFeat_fd_ROI)
                    if feat == "right_eye":
                        #print("MAIN AT right_eye: " + feat)
                        result_RightEye.append(ROI_Returned)
                        result_RightEye_HOG.append(PyFeat_ROI_hog_image)
                        result_RightEye_HOG_fd.append(PyFeat_fd_ROI)
                    if feat == "left_eye":
                        #print("MAIN AT left_eye: " + feat)
                        result_LeftEye.append(ROI_Returned)
                        result_LeftEye_HOG.append(PyFeat_ROI_hog_image)
                        result_LeftEye_HOG_fd.append(PyFeat_fd_ROI)
    
            #Add Image Filename to output --------------------------------------------------
                Face_AU['Filename'] = fileName
                #PyFeat_hog_image['Filename'] = fileName
                #PyFeat_fd['Filename'] = fileName
                Face_Coords_XY['Filename'] = fileName
                Face_Emotion['Filename'] = fileName
    
            #Add Results to the Master Tables: --------------------------------------------------
                result_AU[j]            = Face_AU               # SET ONE: Action Units: AU
                result_HOG[j]           = PyFeat_hog_image      # SET TWO: Histogram of Gradients (HOG)
                result_FaceDim[j]       = PyFeat_fd             # SET THREE: Facial Dimensions
                result_FaceLandmark[j]  = Face_Coords_XY        # SET FOUR: Face X, Y Landmarks
                result_Emotion[j]       = Face_Emotion          # SET FIVE: Emotions
    
                result_INDEX[j]         = fileName         # SET SIX: Cropped Image
    
                j = j + 1
       
          
                result_Images.append(image_Cropped)
    
    
            # CREATE THE OUTPUT PLOTS FOR ANALYSIS --------------------------------------------------
                if outPlots == True:
                    fig, (ax1, ax2,ax3, ax5, ax4,ax6) = plt.subplots(nrows = 1,ncols = 6, figsize=(16, 32), sharex=False, sharey=False)
                    #fig, (ax4) = plt.subplots(nrows = 1,ncols = 1, figsize=(4, 8), sharex=False, sharey=False)
                    ax1.axis('off')
                    ax1.imshow(OrigImage, cmap=plt.cm.gray)
                    ax1.set_title('Original Image')
                    
                    ax2.axis('off')
                    ax2.imshow(image_Cropped, cmap=plt.cm.gray)
                    ax2.set_title('Cropped Image')
                    
                    # Rescale histogram for better display
                    hog_image_rescaled = exposure.rescale_intensity(PyFeat_hog_image, in_range=(0, 10))
                    ax3.axis('off')
                    ax3.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                    ax3.set_title('Histogram of Oriented Gradients')
        
                    #aspectRatio = 1/ax3.get_data_ratio()
                    aspectRatio = 1/8
                    
                    ax5.axis('off')
                    ax5.set_aspect(1/ax3.get_data_ratio(), adjustable = 'box')
                    ax5.scatter(faceONE["X"],-faceONE["Y"])
                    ax5.set_title('Landmarks')
        
                    
                    ax4.set_aspect(aspectRatio, adjustable = 'box')
                    ax4.barh(Plot_emotion,Plot_score)
                    ax4.set_title('Emotions')
        
                    ax6.set_aspect(aspectRatio, adjustable = 'box')
                    ax6.barh(Plot_ActionUnit,AU_Plot_score)
                    ax6.set_title('Action Units')
        
        
                    plt.rcParams['figure.dpi'] = dpi
                    plt.show()
                    
  
            else:
                print("NO CROPPING :(")
    
        else:
            print("NoFrames")
            success = False
            cv2.destroyAllWindows()



# -----------------------------------------------------------------------------------------------------------------------------



#Flatten Action Units
flat_AU = pd.concat(result_AU, axis=0, keys=None)
flat_AU.insert(0, 'ImageID', range(0, len(flat_AU)))

flat_Emotion = pd.concat(result_Emotion, axis=0, keys=None)
flat_Emotion.insert(0, 'ImageID', range(0, len(flat_AU)))

flat_FaceLandmark = pd.concat(result_FaceLandmark, axis=0, keys=None)
flat_FaceLandmark.insert(0, 'ImageID', range(0, len(flat_AU)))




# ------------------------------------------------------------------------------------------------------------------------
# CORRELATION STUDIES


# Correlation Plot of Action Units
CorrelatedAU = flat_AU
CorrelatedAU = CorrelatedAU.drop("ImageID", axis=1, inplace=False)
CorrelatedAU = CorrelatedAU.drop("Filename", axis=1, inplace=False)
CorrelatedAU = CorrelatedAU.drop("FaceID", axis=1, inplace=False)

# Original Correlation
# Plot the Correlation
plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(flat_AU.corr(), dtype=np.bool))
heatmap = sns.heatmap(flat_AU.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':20}, pad=16);
CorrelatedAU = CorrelatedAU.corr()

# Correlation Plot of Facial Landmarks
CorrelatedFaceLandmark = flat_FaceLandmark
CorrelatedFaceLandmark = CorrelatedFaceLandmark.drop("ImageID", axis=1, inplace=False)
CorrelatedFaceLandmark = CorrelatedFaceLandmark.drop("Filename", axis=1, inplace=False)
CorrelatedFaceLandmark = CorrelatedFaceLandmark.corr()


# --------------------------------------------------------------------------------------------------------------
#Save Results

# Current Path

currProj = "OutFinal"
currRoot = "F:\Project_FacialReenactmentOutputs\Actor_12"
currdate = datetime.datetime.now(pytz.timezone('US/Eastern'))

newDir = "FacesOut_" + str(currdate.year) + "_" + str(("00" + str(currdate.month))[-2:]) + "_" + str(("00" + str(currdate.day))[-2:]) + "_HR_" +  str(("00" + str(currdate.hour))[-2:] +  str(("00" + str(currdate.minute))[-2:]) + "_" + currProj)



# Create HOME Directory for Study
DirToCreate = currRoot + "/" + newDir + "/"  
#Path(DirToCreate).mkdir(parents=True, exist_ok=True)
os.mkdir(DirToCreate)


# Create File Directory for Key Files
DirToCreate = currRoot + "/" + newDir + "/SummaryAndData/"  
#Path(DirToCreate).mkdir(parents=True, exist_ok=True)
os.mkdir(DirToCreate)
# File Summaries:

flat_AU.to_csv(DirToCreate+"AU_OutputData.csv")
flat_Emotion.to_csv(DirToCreate+"Emotion_OutputData.csv")
flat_FaceLandmark.to_csv(DirToCreate+"FaceLandmark_OutputData.csv")


#Write Index Dictionary to Disk
fileString = "INDEX.csv"
dictToCSV(DirToCreate,fileString, result_INDEX)


#Write fileDetails to Disk
fileDetails.to_csv(DirToCreate+"VideoFileDetails.csv")

#Write Correlated Studies to Disk
CorrelatedAU.to_csv(DirToCreate+"AU_CorrelationResults.csv")
CorrelatedFaceLandmark.to_csv(DirToCreate+"FacialLandmark_CorrelationResults.csv")




# ---------------------------------------------------------------------------------------------------------------------------
# BUILD MONTAGES AND SAVE TO DISK

# Current Path

# Create File Directory for Key Files
DirToCreate = currRoot + "/" + newDir + "/Montages_RAW/"  
#Path(DirToCreate).mkdir(parents=True, exist_ok=True)
os.mkdir(DirToCreate)

#Montage Visualization

frameWidth   = 125
frameHeight  = 110
montageCols  = 20
montageRows  = 20


dataFacePart = result_RightEye
Path         = DirToCreate
FacePart     = "result_RightEye"
# Save Montage
Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster = 99 )


dataFacePart = result_LeftEye
Path         = DirToCreate
FacePart     = "result_LeftEye"
Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster = 99 )


# Create Right Eye Montage and Save to Disk
dataFacePart = result_Mouth
Path         = DirToCreate
FacePart     = "result_Mouth"
# Save Montage
Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster = 99 )


# Create Right Eye Montage and Save to Disk
dataFacePart = result_Images
Path         = DirToCreate
FacePart     = "FullCroppedFace"
# Save Montage
Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster = 99 )

# -----------------------------------------------------------------------------------------------------------------------------

# SAVE IMAGES TO DISK

# Create Directory for Raw Cropped Images
DirToCreate = str(currRoot + "/" + newDir + "/Images_RAW_CROPPED/"  )
os.mkdir(DirToCreate)
Root = DirToCreate

# FULL FACE
FacePart = "FullFace"
listOfFileNames, DirToCreate = Save_FilesToSingleDir_NoCluster(result_Images, Root, FacePart)


# Left Eye
FacePart = "LeftEye"
listOfFileNames, DirToCreate = Save_FilesToSingleDir_NoCluster(result_LeftEye, Root, FacePart)


# Right Eye
FacePart = "RightEye"
listOfFileNames, DirToCreate = Save_FilesToSingleDir_NoCluster(result_RightEye, Root, FacePart)

# Mouth
FacePart = "Mouth"
listOfFileNames, DirToCreate = Save_FilesToSingleDir_NoCluster(result_Mouth, Root, FacePart)


# -----------------------------------------------------------------------------------------------------------------------------

# SAVE HOG DATA TO DISK

# Create Directory for Raw Cropped Images
DirToCreate = str(currRoot + "/" + newDir + "/HOG_IMAGE/"  )
os.mkdir(DirToCreate)
Root = DirToCreate


# Raw HOG Images
with open(DirToCreate+'LeftEye_HOG_Images.pkl', 'wb') as f:
    pickle.dump(result_LeftEye_HOG, f)


with open(DirToCreate+'RightEye_HOG_Images.pkl', 'wb') as f:
    pickle.dump(result_RightEye_HOG, f)
    
with open(DirToCreate+'Mouth_HOG_Images.pkl', 'wb') as f:
    pickle.dump(result_Mouth_HOG, f)

# HOG Data
DirToCreate = str(currRoot + "/" + newDir + "/HOG_DATA/"  )
os.mkdir(DirToCreate)
Root = DirToCreate
    
# HOG Data


# Code to Flatten List of Numpy Arrays - SINGLE LIST
result_LeftEye_HOG_fd_FLAT = np.stack(result_LeftEye_HOG_fd, axis = 0)
result_LeftEye_HOG_fd_FLAT.insert(0, 'ImageID', range(0, len(result_LeftEye_HOG_fd_FLAT)))

result_RightEye_HOG_fd_FLAT = np.stack(result_RightEye_HOG_fd, axis = 0)
result_RightEye_HOG_fd_FLAT.insert(0, 'ImageID', range(0, len(result_RightEye_HOG_fd_FLAT)))

result_Mouth_HOG_fd_FLAT   = np.stack(result_Mouth_HOG_fd, axis = 0)
result_Mouth_HOG_fd_FLAT.insert(0, 'ImageID', range(0, len(result_Mouth_HOG_fd_FLAT)))


with open(DirToCreate+'LeftEye_HOG_Data.pkl', 'wb') as f:
    pickle.dump(result_LeftEye_HOG_fd_FLAT, f)


with open(DirToCreate+'RightEye_HOG_Data.pkl', 'wb') as f:
    pickle.dump(result_RightEye_HOG_fd_FLAT, f)
    
with open(DirToCreate+'Mouth_HOG_Data.pkl', 'wb') as f:
    pickle.dump(result_Mouth_HOG_fd_FLAT, f)
    


# Save details of how the HOG was generated:
    
    


# THIS CONCLUDES THE FACIAL PARSER  ANY WORK BEYOND THIS IS THE FACIAL PALETTE INVOLVING CLUSTERING AND SSIM CALCULATIONS

# END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------







# USE THIS IN THE CORRELATION STUDIES:


# ----------------------------------------------------------------------------------------------------------
# Reduce Action Units:  https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6
# Needed a holistic way to drop columns


# Determine which columns to drop:
def corrX_new(df, cut = 0.9) :
       
    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))
    
    dropcols = list()
    
    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                 'v2.target','corr', 'drop' ]))
    
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else: 
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]
                
                s = pd.Series([ corr_mtx.index[row],
                up.columns[col],
                avg_corr[row],
                avg_corr[col],
                up.iloc[row,col],
                drop],
                index = res.columns)
        
                res = res.append(s, ignore_index = True)
    
    dropcols_names = calcDrop(res)
    
    return(dropcols_names)


def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
    
    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))
     
    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop 
    poss_drop = list(set(poss_drop).difference(set(drop)))
    
    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        
    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)
         
    return drop



drop_new = corrX_new(CorrelatedAU, cut = 0.9)
len(drop_new)
# Out[247]: 194
drop_new

# Correlation Plot of Action Units

CorrelatedAU.drop(drop_new, axis=1, inplace=True)

# Original Correlation
# Plot the Correlation
plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(CorrelatedAU.corr(), dtype=np.bool))
heatmap = sns.heatmap(CorrelatedAU.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':20}, pad=16);










# -----------------------------------------------------------------------------------------------------------------------------

















# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------
#  Save MOUTH Images
#  Save AU List to CSV - Manipulate in Excel for Proof of Concpet
#  Saves the EYE IMAGES instead of the whole face.
#----------------------------------------------------------------------------------------------

studyPath = "f:/Project_FacialReenactmentOutputs/"
result_HOG = result_LeftEye_HOG_fd_FLAT #, result_RightEye_HOG_fd_FLAT, result_Mouth_HOG_fd_FLAT
FacePart = ["result_LeftEye","result_RightEye","result_Mouth"]
FacePart = FacePart[0]

# -----------------------------------------------------------------------------------------------
# Fast KMeans Trials

kmeans = KMeans(n_clusters=19, mode='euclidean', verbose=1)

kmeans.__getattribute__()
x_data = torch.from_numpy(result_Mouth_HOG_fd_FLAT)

labels_Clust = kmeans.fit_predict(x_data.float())

out_Labels   = labels_Clust.to("cuda")
out_Labels   = out_Labels.to("cpu").numpy()
out_Labels.tolist()

#----------------------------------------------------------------------------------------------

# Save Each File to a Distinct Directory based on CLUSTER
studyPath = "f:/Project_FacialReenactmentOutputs/"
Save_FileToClusterDirctory(out_Labels,result_Mouth,CurrentActor,studyPath,FacePart)


# Save Each File to a SINGLE Directory CLUSTER with Cluster ID 

studyPath = "f:/Project_FacialReenactmentOutputs/"
listOfFiles, listOfClusters, directoryOfImages = Save_FilesToSingleDirctory(out_Labels,result_Mouth,CurrentActor,studyPath, FacePart, SaveCSV = True)


# Clean up File Name
fileNameIterator = map(lambda fileName: os.path.basename(fileName), listOfFiles)
listOfFilesNames = (list(map(lambda fileName: os.path.basename(fileName), listOfFiles)))
# listOfFiles = [os.path.splitext(x)[0] for x in listOfFilesNames]


#Build CSV - FinalPath, File Name, Cluster
list_of_Tuples = list(zip(listOfFiles, listOfFilesNames, listOfClusters))
df_ImageToCluster = pd.DataFrame(list_of_Tuples,columns = ['FileAndPath', 'File Name', 'Class Label'])
df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster.csv", index = False)

df_ImageToCluster = df_ImageToCluster[[ 'Class Label', 'File Name']]
df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster_Formatted.csv", index = False)


# Save Action Units:
   
#Add Image ID to the flagAU file:
flat_AU.to_csv(directoryOfImages+"df_ImageToActionUnit.csv", index = False)



# ---------------------------------------------------------------------------------------------------------------------------
















# -----------------------------------------------------------------------------------------------
#  Save Eye Images
#  Save AU List to CSV - Manipulate in Excel for Proof of Concpet
#  Saves the EYE IMAGES instead of the whole face.
#----------------------------------------------------------------------------------------------

# Preview Facial Images - QA Check:
# Create Right Eye Montage and Save to Disk
dataFacePart = result_RightEye
frameWidth   = 250
frameHeight  = 120
montageCols  = 24
montageRows  = 14
#Path         = None #"f:"
Path         = "f:"
FacePart     = "result_RightEye"
# Save Montage
Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster = 99 )



# -----------------------------------------------------------------------------------------------
# Fast KMeans Trials

kmeans = KMeans(n_clusters=20, mode='euclidean', verbose=1)
x_data = torch.from_numpy(result_RightEye_HOG_fd_FLAT)

labels_Clust = kmeans.fit_predict(x_data.float())

out_Labels   = labels_Clust.to("cuda")
out_Labels   = out_Labels.to("cpu").numpy()
out_Labels.tolist()

#----------------------------------------------------------------------------------------------
# Save Results to Disk - For Dataloader trials

# Save Each File to a Distinct Directory based on CLUSTER
studyPath = "f:/Project_FacialReenactmentOutputs/"

Save_FileToClusterDirctory(out_Labels,result_RightEye,CurrentActor,studyPath,FacePart)


# Save Each File to a SINGLE Directory CLUSTER with Cluster ID 

studyPath = "f:/Project_FacialReenactmentOutputs/"

listOfFiles, listOfClusters, directoryOfImages = Save_FilesToSingleDirctory(out_Labels,result_RightEye,CurrentActor,studyPath,FacePart, SaveCSV = True)


# Clean up File Name
fileNameIterator = map(lambda fileName: os.path.basename(fileName), listOfFiles)

listOfFilesNames = (list(map(lambda fileName: os.path.basename(fileName), listOfFiles)))
# listOfFiles = [os.path.splitext(x)[0] for x in listOfFilesNames]


#Build CSV - FinalPath, File Name, Cluster

list_of_Tuples = list(zip(listOfFiles, listOfFilesNames, listOfClusters))
df_ImageToCluster = pd.DataFrame(list_of_Tuples,columns = ['FileAndPath', 'File Name', 'Class Label'])
df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster.csv", index = False)

df_ImageToCluster = df_ImageToCluster[[ 'Class Label', 'File Name']]
df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster_Formatted.csv", index = False)


# Save Action Units:
    
flat_AU.to_csv(directoryOfImages+"df_ImageToActionUnit.csv", index = False)


# ---------------------------------------------------------------------------------------------------------------------------






# -----------------------------------------------------------------------------------------------
#  Save Left Eye Images
#  Save AU List to CSV - Manipulate in Excel for Proof of Concpet
#  Saves the EYE IMAGES instead of the whole face.
#----------------------------------------------------------------------------------------------

# Preview Facial Images - QA Check:
# Create Right Eye Montage and Save to Disk
dataFacePart = result_LeftEye
frameWidth   = 250
frameHeight  = 120
montageCols  = 24
montageRows  = 14
#Path         = None #"f:"
Path         = "f:"
FacePart     = "result_LeftEye"
# Save Montage
Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster = 99 )



# -----------------------------------------------------------------------------------------------
# Fast KMeans Trials

kmeans = KMeans(n_clusters=20, mode='euclidean', verbose=1)
x_data = torch.from_numpy(result_RightEye_HOG_fd_FLAT)

labels_Clust = kmeans.fit_predict(x_data.float())

out_Labels   = labels_Clust.to("cuda")
out_Labels   = out_Labels.to("cpu").numpy()
out_Labels.tolist()


kmeans.minibatch
#----------------------------------------------------------------------------------------------


# Did not work well - ran out of memory:
# https://github.com/subhadarship/kmeans_pytorch
    
from kmeans_pytorch import kmeans, kmeans_predict

# set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# data
data_size, dims, num_clusters = 1000, 2, 18


# k-means
cluster_ids_x, cluster_centers = kmeans(
    X=x_data, num_clusters=num_clusters, distance='euclidean', device=device
)



out_Labels_V2   = cluster_ids_x.to("cuda")
out_Labels_V2   = out_Labels_V2.to("cpu").numpy()
out_Labels_V2.tolist()

#--------------------------------------------------------------------------------------------


# Again, taken from: 
# https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2


ks = range(1, 2)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(x_data)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


#----------------------------------------------------------------------------------------------
# Save Results to Disk - For Dataloader trials

# Save Each File to a Distinct Directory based on CLUSTER
studyPath = "f:/Project_FacialReenactmentOutputs/"

Save_FileToClusterDirctory(out_Labels,result_RightEye,CurrentActor,studyPath,FacePart)


# Save Each File to a SINGLE Directory CLUSTER with Cluster ID 

studyPath = "f:/Project_FacialReenactmentOutputs/"

listOfFiles, listOfClusters, directoryOfImages = Save_FilesToSingleDirctory(out_Labels,result_RightEye,CurrentActor,studyPath,FacePart, SaveCSV = True)


# Clean up File Name
fileNameIterator = map(lambda fileName: os.path.basename(fileName), listOfFiles)

listOfFilesNames = (list(map(lambda fileName: os.path.basename(fileName), listOfFiles)))
# listOfFiles = [os.path.splitext(x)[0] for x in listOfFilesNames]


#Build CSV - FinalPath, File Name, Cluster

list_of_Tuples = list(zip(listOfFiles, listOfFilesNames, listOfClusters))
df_ImageToCluster = pd.DataFrame(list_of_Tuples,columns = ['FileAndPath', 'File Name', 'Class Label'])
df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster.csv", index = False)

df_ImageToCluster = df_ImageToCluster[[ 'Class Label', 'File Name']]
df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster_Formatted.csv", index = False)


# Save Action Units:
    
flat_AU.to_csv(directoryOfImages+"df_ImageToActionUnit.csv", index = False)


# ---------------------------------------------------------------------------------------------------------------------------


# END HERE HERE HERE EHEREHEEREREHEREHEHEHRERHERHERHERHERHERH

# ---------------------------------------------------------------------------------------------------------------------------


# Mouth BASED Clustering - ORIGINAL ---------------------------------------------------------------------------------------------------------------------------
# Clusters the Larger Complete Face based on cluster of the mouth
# SAVES all the full face images.
# -----------------------------------------------------------------------------------------------
# Fast KMeans Trials



kmeans = KMeans(n_clusters=20, mode='euclidean', verbose=1)

x_data = torch.from_numpy(result_Mouth_HOG_fd_FLAT)

labels_Clust = kmeans.fit_predict(x_data.float())

out_Labels   = labels_Clust.to("cuda")
out_Labels   = out_Labels.to("cpu").numpy()
out_Labels.tolist()

#----------------------------------------------------------------------------------------------
# Save Results to Disk - For Dataloader trials

# Save Each File to a Distinct Directory based on CLUSTER
studyPath = "f:/Project_FacialReenactmentOutputs/"

Save_FileToClusterDirctory(out_Labels,result_Images,CurrentActor,studyPath)


# Save Each File to a SINGLE Directory CLUSTER with Cluster ID 

studyPath = "f:/Project_FacialReenactmentOutputs/"

listOfFiles, listOfClusters, directoryOfImages = Save_FilesToSingleDirctory(out_Labels,result_Images,CurrentActor,studyPath, SaveCSV = True)


# Clean up File Name
fileNameIterator = map(lambda fileName: os.path.basename(fileName), listOfFiles)

listOfFilesNames = (list(map(lambda fileName: os.path.basename(fileName), listOfFiles)))
# listOfFiles = [os.path.splitext(x)[0] for x in listOfFilesNames]


#Build CSV - FinalPath, File Name, Cluster

list_of_Tuples = list(zip(listOfFiles, listOfFilesNames, listOfClusters))
df_ImageToCluster = pd.DataFrame(list_of_Tuples,columns = ['FileAndPath', 'Filename', 'Cluster'])

df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster.csv", index = False)

# ---------------------------------------------------------------------------------------------------------------------------































# DATASET AND DATALOADER Trial -------------------------------------------------------



# Custome Dataloader example: https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L09/code/custom-dataloader/custom-dataloader-example.ipynb

# Great walkthrough: https://blog.paperspace.com/dataloaders-abstractions-pytorch/




# transforms.Grayscale(num_output_channels=1)


class MyDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]



from torchvision import transforms
from torch.utils.data import DataLoader


# Note that transforms.ToTensor()
# already divides pixels by 255. internally

custom_transform = transforms.Compose([#transforms.Lambda(lambda x: x/255.), # not necessary
                                       transforms.ToTensor(),
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.Resize(28),
                                      ])

train_dataset = MyDataset(csv_path=directoryOfImages+'df_ImageToCluster_Formatted.csv',
                          img_dir=directoryOfImages,
                          transform=custom_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          drop_last=True,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0) # number processes/CPUs to use


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

num_epochs = 2
for epoch in range(num_epochs):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)



print(x.shape)


x_image_as_vector = x.view(-1, 28*28)
print(x_image_as_vector.shape)





# One Hot Encode Clusters


# binary encode

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(listOfClusters)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)






# Images and Cluster Labels -------------------------------------------------------------------   

tensor_out_Labels = torch.tensor(out_Labels)
clusterOneHot = torch.nn.functional.one_hot(tensor_out_Labels)

# Convert list of appended images to PyTorch array

test_Result_Images = result_Images[0:4]

tensor_resultImages = torch.as_tensor(test_Result_Images)





























# Iterate through Dataset ------------------------------------------------------------
        
    
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break
    



# WORK IN PROGRESS --------------------------------------------------------------------    
    

# Path = "F:"

# dataFacePart = ImagesFromCluster
# frameWidth   = 120
# frameHeight  = 80
# montageCols  = 16
# montageRows  = 12
# Path         = None #"f:"
# Cluster      = currCluster
# #Path         = "f:"

# FacePart     = "result_Face"



# #Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart, Cluster )


# listIndex = FileNameFromCluster.Index.tolist()
# print(FileNameFromCluster)




cv2.destroyAllWindows()



# ---------------------------------------------------------------------------------------------------------------------------------



# Montage of faces:  Montage Visualization


ImagesFromCluster = imagesToShow.loc[outClusters['Cluster']==2,'Image']

Path = "F:"

dataFacePart = ImagesFromCluster
frameWidth   = 120
frameHeight  = 80
montageCols  = 16
montageRows  = 12
Path         = None #"f:"
#Path         = "f:"

FacePart     = "result_Face"



Montage_FacePart ( dataFacePart, frameWidth, frameHeight, montageCols, montageRows, Path, FacePart )

































# ------------------------------------------------------------------------------------------

print("HOG MOUTH - result_LeftEye_HOG")
print("HOG Matrix Labels: ", len(result_LeftEye_HOG_fd_FLAT))
print("HOG EYE")
print("HOG Matrix Labels: ", len(result_LeftEye_HOG_fd_FLAT))
print("Number of Files Recorded: ", j)

      

      
from sklearn.decomposition import PCA
pcaComponents = len(result_LeftEye_HOG_fd_FLAT)
pca = PCA(n_components=pcaComponents)
pca.fit(result_LeftEye_HOG_fd_FLAT)


import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');













# ------------------------------------------------------------------------------------------

# PCA Work





explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
plt.plot(np.cumsum(explained_variance))
print(np.sum(explained_variance), np.sum(explained_variance_ratio))

# Use the following for the number of features:

pcaComponents = np.sum(explained_variance)
pcaComponents = int(round(pcaComponents,0))

if pcaComponents > len(result_LeftEye_HOG_fd_FLAT):
    pcaComponents = len(result_LeftEye_HOG_fd_FLAT)-1

print("PCA Components to be used: ",pcaComponents)

#pcaComponents = 1000  #OVERRIDE PCVALUE HERE

pca = PCA(n_components=pcaComponents)


    


pca.fit(result_LeftEye_HOG_fd_FLAT)

data_for_PCA_test = pca.transform(result_LeftEye_HOG_fd_FLAT)
print("Rows of Data for PCA Test: ", len(data_for_PCA_test))
data_for_PCA_test.shape


# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

k_means_DataIn = data_for_PCA_test
#k-means_DataIn = HOG_Matrix
n_components = 30

# Create a PCA instance: pca
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(k_means_DataIn)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)



plt.scatter(PCA_components[1], PCA_components[2], alpha=.1, color='black')
plt.xlabel('PCA 1')

plt.ylabel('PCA 2')



#Scaling
from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()
#data = pd.DataFrame(scaler.fit_transform(HOG_Matrix), columns=HOG_Matrix.columns)

#PCA Transformation
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(result_LeftEye_HOG_fd_FLAT)
PCAdf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2','principal component 3'])

datapoints = PCAdf.values
m, f = datapoints.shape
k = 3







from mpl_toolkits.mplot3d import Axes3D



#Visualization
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = datapoints
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("principal component 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("principal component 1")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("principal component 1")
ax.w_zaxis.set_ticklabels([])
plt.show()

def init_medoids(X, k):
    from numpy.random import choice
    from numpy.random import seed
 
    seed(1)
    samples = choice(len(X), size=k, replace=False)
    return X[samples, :]

medoids_initial = init_medoids(datapoints, 3)



# Again, taken from: 
# https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2


ks = range(1, 25)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()



# Using k = 4, we will determine which cluster each facial expression lives:
clusters = 24
model_KMeans = KMeans(n_clusters=clusters)
model_KMeans.fit(PCA_components)



#model_KMeans.labels_
clusters =  model_KMeans.labels_
print(type(clusters))

print("KMeans Labels: ", len(model_KMeans.labels_))
print("Number of Files Recorded: ", len(files))


rng = np.clusters.RandomState(10)





np.histogram(clusters, bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26])







#Build new output

filesTest = np.array(result_INDEX)
filesTest['cluster'] = clusters
#
#HOG_Matrix,files


# TRIAL - CLUSTER ONE
hist2 = filesTest.cluster.hist()#(bins = clusters)








cluster16 = filesTest.Path[(filesTest.cluster == 25 )]
# initialize the list of images
imagePaths = cluster16
print(type(imagePaths))
imagePaths = imagePaths.astype(str)
images = []
imageTots = 0
# loop over the list of image paths
for imagePath in imagePaths:
    imageTots = imageTots + 1
    #print("Total Images So far:" , imageTots)
    #print("Current Path ", imagePath)
    #print(imagePath[2:3])
    if imagePath[2:4] == "F:":
        imagePath = imagePath[2:len(imagePath)-2]
        
   # print("Current Image Path: ", glob.escape(imagePath))
    image = cv2.imread(imagePath)
    #imgplot = plt.imshow(image)
    #plt.show()
    images.append(image)
    
    
## construct the montages for the images

montages = build_montages(images, (128, 196), (14, 5))

## loop over the montages and display each of them
for montage in montages:
    print(type(montage))
    cv2.imshow("Montage", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



























# -----------------------------------------------------------------------------------------------------------------------------



import math

allLen = []

for currHog in result_Mouth_HOG:
    currLen = len(currHog)
    allLen.append(currLen) 
    
    max(allLen)
    min(allLen)


for currHog in result_LeftEye_HOG:
    currLen = len(currHog)
    allLen.append(currLen) 
    
    max(allLen)
    min(allLen)


bins = np.linspace(math.ceil(min(allLen)), 
                   math.floor(max(allLen)),
                   20) # fixed number of bins

plt.xlim([min(allLen)-5, max(allLen)+5])

plt.hist(allLen, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed number of bins)')
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()




# -----------------------------------------------------------------------------------------------------------------------------


# Similarity Matrix


import torch
import time
useTorch = True

num = len(result_Mouth_HOG)
similarity_matrix = np.zeros((num, num))



HOG_Matrix_torch = np.concatenate(result_Mouth_HOG)
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


# --------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------------




k = 50
medoids = []
cluster_files = []
for i in range(k):
    currentIndex = (int(indices[i]))
    print("Current i: ", i, "  CurrentIndex: ", currentIndex)
    medoids.append(result_Mouth_HOG[indices[i]])
    cluster_files.append(files_FULL.Path.iloc[currentIndex])
    
medoids = np.asarray(medoids)
#cluster_files = np.asarray(cluster_files)

files_FULL


# -----------------------------------------------------------------------------------------------------------------------------








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













# plt.axis("off")
# plt.imshow(result_LeftEye_HOG[40])
# plt.show()



# result_LeftEye_HOG[42]








# # result_Mouth, result_RightBrow, result_LeftBrow, result_RightEye, result_LeftEye
                   


# Verbose = False
# Path = fileName
# file

















# # # Condense data:
    
# # result_Emotion = pd.concat(result_Emotion)
# # result_AU = pd.concat(result_AU)
# # #result_FaceDim
# # result_FaceLandmark = pd.concat(result_FaceLandmark)
# # #result_Emotion = pd.concat(result_Emotion)
# # #result_INDEX


# # Cluster Results















# # ROI - Region of Interest Generator -------------------------------------------------------------------------------
# def Alt_ImageROI (imageIn, LandmarkData, FacialLandmark, Verbose = True):

#     AllowedFeatures = ["mouth","right_eyebrow","left_eyebrow", "right_eye","left_eye","nose", "jaw" ]


        
#     FACIAL_LANDMARKS_IDXS = [
#     	("alt_mouth", [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,32,33,34,35,36]),
#     	("alt_right_eye", [18,19,20,21,22,41,42]),
#     	("alt_left_eye",[23,24,25,26,27,47,48])
#     ]

# FACIAL_LANDMARKS_IDXS["alt_mouth"]


#     featureAllowed = FacialLandmark in AllowedFeatures

#     if featureAllowed:
#         (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS[FacialLandmark]
        
#         rStart = int(rStart)
#         rEnd   = int(rEnd)
        
#     # loop over the subset of facial landmarks, drawing the
#       		# specific face part
#         # for (x, y) in LandmarkData[rStart:rEnd]:
#         #     print("X:",x,"  Y",y)
#         #     cv2.circle(imageIn, (int(x), int(y)), 1, (0, 0, 255), -1)
     
#     # # extract the ROI of the face region as a separate image
#         (x, y, w, h) = cv2.boundingRect(np.array([LandmarkData[rStart:rEnd]]))
#         x = x
#         y = y 
#         h = h
#         roi = imageIn[y:y + h, x:x + w]
#         roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    		
#     # # # show the particular face part
    
#         if Verbose == False:
#             currFeat = "ROI: " + str(FacialLandmark)
#             plt_imshow(currFeat, roi)
     
#     return roi





















# OLD CODE -------------------------------------------------------------------------------------------------------------------



            
            
#             #plt.barh(Plot_ActionUnit,AU_Plot_score)

# #             # EYE LANDMARKS 
# #             # Taken from https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/?_ga=2.201225262.1858633918.1635734838-655960416.1621557650
# #             # grab the indexes of the facial landmarks for the left and
# #             # right eye, respectively


# # MOUTH CODE:

#             (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#             (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#             (mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#             i = int(mouthStart)
#             j = int(mouthEnd)

#             np_faceONE = faceONE.to_numpy()
#             np_faceONE = np_faceONE.astype(int)            

#             clone = image_Cropped.copy()
#             #plt.imshow(image_Cropped)

#         # loop over the subset of facial landmarks, drawing the
#   		# specific face part
#             for (x, y) in np_faceONE[i:j]:
#                 print("X:",x,"  Y",y)
#                 cv2.circle(clone, (int(x), int(y)), 1, (0, 0, 255), -1)
 
#         # # extract the ROI of the face region as a separate image
#             (x, y, w, h) = cv2.boundingRect(np.array([np_faceONE[i:j]]))
#             x = x
#             y = y 
#             h = h
#             roi = image_Cropped[y:y + h, x:x + w]
#             roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        		
#         # # show the particular face part
#             plt_imshow("ROI", roi)
#             plt_imshow("Image", clone)





# # # EYE CODE:

#             #(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# #(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# i = int(rStart)
# j = int(rEnd)

# np_faceONE = faceONE.to_numpy()
# np_faceONE = np_faceONE.astype(int)            

# clone = image_Cropped.copy()
# #plt.imshow(image_Cropped)

# # loop over the subset of facial landmarks, drawing the
#   		# specific face part
# for (x, y) in np_faceONE[i:j]:
#     print("X:",x,"  Y",y)
#     cv2.circle(clone, (int(x), int(y)), 1, (0, 0, 255), -1)
     
# # # extract the ROI of the face region as a separate image
# (x, y, w, h) = cv2.boundingRect(np.array([np_faceONE[i:j]]))
# x = x
# y = y 
# h = h
# roi = image_Cropped[y:y + h, x:x + w]
# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
# 		
# # # show the particular face part
# plt_imshow("ROI", roi)
# plt_imshow("Image", clone)



# eyes= eye_detect.detectMultiScale(rol_grey)

# for (ex,ey,ew,eh) in eyes:
#     cv2.rectangle(rol_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
#     cv2.imshow('Img',img)