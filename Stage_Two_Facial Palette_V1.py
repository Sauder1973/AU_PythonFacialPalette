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


from fast_pytorch_kmeans import KMeans
import torch
from torch.utils.data import Dataset

import datetime
import pytz

from SaveImages import Save_FileToClusterDirctory
from SaveImages import Save_FilesToSingleDirctory
from SaveImages import Save_FilesToSingleDir_NoCluster


from imutils import build_montages
from imutils import paths

import glob
from sys import argv

import sewar as simu
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from skimage.metrics import structural_similarity as ssim2


# -------------------------------------------------------------------------------------------------------------------------------------------
# Constants


# Setup HOG Modeling
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"

detector = featDetect(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)


Actor = "Actor_12"
StudyPath = "FacesOut_2022_02_26_HR_2049_OutFinal"
RootPath = LeftEyeImagesPath = "F:\Project_FacialReenactmentOutputs"



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
            df_tmp = {'File Name': fileToAnalyze,
                        'File': file,
                        'FacialFeature':FacialFeature[1]}            
            fileDetails.append(df_tmp)
            
            #print(fileToAnalyze)
            filesForAnalysis.append(fileToAnalyze)


images_FacialPart = pd.DataFrame(fileDetails)
images_FacialPartFilesWithPath = images_FacialPart[["File Name"]]


# Open PKL File - HOG numpy Array from Stage One: Facial Parser

pickleHOG_Root = 'F://Project_FacialReenactmentOutputs//Actor_12//FacesOut_2022_02_26_HR_2049_OutFinal//HOG_DATA//'

with open(pickleHOG_Root+'LeftEye_HOG_Data.pkl', 'rb') as f:
    result_LeftEye_HOG_fd_FLAT = pickle.load(f)

# -----------------------------------------------------------------------------------------------------------------------------------------------



# Fast KMeans Trials

clusters = 20
kmeans = KMeans(n_clusters=clusters, mode='euclidean', verbose=1)
x_data = torch.from_numpy(result_LeftEye_HOG_fd_FLAT)

labels_Clust = kmeans.fit_predict(x_data.float())

out_Labels   = labels_Clust.to("cuda")
out_Labels   = out_Labels.to("cpu").numpy()
out_Labels   = out_Labels.tolist()



# Visualize and/or Save results of clustering

# Histograms of each cluster:


# fixed bin size
bins = np.arange(0, clusters, 1) # fixed bin size
plt.xlim([min(out_Labels), max(out_Labels)+5])
plt.title('Number of Images by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Images')
plt.hist(out_Labels, bins=bins, alpha=0.5)
plt.show()



# Qualitative Summaries:
# See Results of Each Cluster:
    
j = 0

uniqueClusters = np.unique(out_Labels)


clusterID = 0

for clusterID in uniqueClusters:
    print(clusterID)

    indices = [idx for idx, x in enumerate(out_Labels) if x == clusterID]

    len(indices)

    clusterImagePaths = images_FacialPart["File Name"][indices]
    
    Header = ("Montage - Cluster: ", clusterID)

    if len(clusterImagePaths) >= 2:
        j = j + 1
        Header = ("Montage - Cluster: ", clusterID," Iteration: ",j)
        clusterImagePaths = clusterImagePaths.astype(str)
        images = []
        imageTots = 0
        # loop over the list of image paths
        for currImagePath in clusterImagePaths:
            imageTots = imageTots + 1
            image = cv.imread(currImagePath)
            images.append(image)

 
        ## construct the montages for the images

        #montages = build_montages(images, (128, 196), (14, 5))
        montages = build_montages(images, (120, 60), (12, 16))



        ## loop over the montages and display each of them
        windows = 1
        for montage in montages:
            winname = "Montage - Cluster: %d --- Iteration: %d" % (clusterID,windows)
            print(winname)
            cv.namedWindow(winname)
            cv.moveWindow(winname,40,30)
            #cv2.imshow("Montage - Cluster: %d --- Iteration: %d" % (i,j), montage)
            cv.imshow(winname,montage)
            cv.waitKey(0)
            cv.destroyAllWindows()
            windows += 1



# Save Results Of Cluster - Summary File with Clustered Images

# Create Parent Path for clusters
RootPath = LeftEyeImagesPath = "F:\Project_FacialReenactmentOutputs"
Actor = "Actor_12"
StudyPath = "FacesOut_2022_02_26_HR_2049_OutFinal"
clustDirRoot = RootPath + "\\" + Actor + "\\" + StudyPath + "\\ClusterResults"
os.mkdir(clustDirRoot)



#Create Directory Structure for Cluster Results
currdate = datetime.datetime.now(pytz.timezone('US/Eastern'))
clustDirChild = "ClusterResults_" + str(currdate.year) + "_" + str(("00" + str(currdate.month))[-2:]) + "_" + str(("00" + str(currdate.day))[-2:]) + "_HR_" +  str(("00" + str(currdate.hour))[-2:] +  str(("00" + str(currdate.minute))[-2:]))

studyPath = clustDirRoot + "\\" +  clustDirChild
os.mkdir(studyPath)


out_FullFilePathToCluster = images_FacialPart#["FullPathToAnalyze"]
#out_FullFilePathToCluster.rename(columns={"FullPathToAnalyze": "File Name"})

out_FullFilePathToCluster["Class Label"] = out_Labels
out_FullFilePathToCluster = out_FullFilePathToCluster[[ 'Class Label', 'File Name']]
out_FullFilePathToCluster.to_csv(studyPath+"\\"+FacialFeature[1]+"_ImageToCluster_Formatted.csv", index = False)




# ---------------------------------------------------------------------------------------------------------------------------


# Cluster Measures:
    
# dirForPics = "F:/Project_FacialReenactmentOutputs/Actor_12/FacesOut_2022_02_11_HR_01_result_LeftEye/"
# dirForPics = "F:/Project_FacialReenactmentOutputs/Actor_12/FacesOut_2022_02_11_HR_01_result_Mouth/"


# dirForPics = "F:/Project_FacialReenactmentOutputs/Actor_12/FacesOut_2022_02_11_HR_01_result_LeftEye/"
# dirForPics = "F:/Project_FacialReenactmentOutputs/Actor_12/FacesOut_2022_02_11_HR_01_result_Mouth/"

dirForPics = RootPath + "\\" + Actor + "\\" + StudyPath +"\\LeftEye_2022_02_26_HR_2055\\"


# Begin Assessing Within Cluster similarities
#Taken From: https://betterprogramming.pub/how-to-measure-image-similarities-in-python-12f1cb2b7281

dataroot = dirForPics
directoryOfImages = dirForPics


directoryToCheck = directoryOfImages +"*/"

directories = glob.glob(directoryToCheck)


resultDictionary = {}

data_dir = directories[0]

directories[16:19]

for data_dir in directories[16:19]:
    
    clusterDictionary = {}
    print(data_dir)
    head, tail = os.path.split(data_dir)
    print("Head: "+ head)
    print("Tail: "+ os.path.basename(head))
    
    CurrDirTail = os.path.basename(head)
    
    
    # # data_dir = directories[10]
    # # #data_dir = dirForPics + "CLUSTER0/"
    
    FileList = os.listdir(data_dir)
    
    # test_path = data_dir + data_dir[1]
    
    # test_img = cv2.imread(test_path)
    
    i = 0
            
    # Actual Implementation
    
    df = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    
    # Change the column names
    df.columns = FileList
      
    # Change the row indexes
    df.index = FileList
    
    # Sewar Methods: https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6
    # PyImage Search: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    
   
    df_SSIM2 = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    
    df_SSIM2.columns = FileList
    df_SSIM2.index = FileList
    

    
    
    imagesCollected = []
    
    #Testfile = FileList[1]
    for Testfile in FileList:
    
        print("Cluster: " + CurrDirTail +"  Primary File: " + Testfile)
        test_path = os.path.join(data_dir, Testfile)
        test_img = cv.imread(test_path, cv.IMREAD_GRAYSCALE)
        
        currImage = test_img.ravel()
        imagesCollected.append(test_img.ravel())
        
        # plt.imshow(test_img, interpolation='nearest')
        # plt.show()
        
        i = 0
        #Curfile = FileList[0]
        
        for Curfile in FileList[i:len(FileList)]:
            
            print("Current File: " + str(i) + " Filename:" +Curfile)
            data_path = os.path.join(data_dir, Curfile)
            data_img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
    
            # plt.imshow(data_img, interpolation='nearest')
            # plt.show()
            
            if data_path != test_path:
                df_SSIM2.loc[Testfile, Curfile] = ssim2(test_img, data_img, multichannel=True)
            else:
                print("Test and Current Image are the same SKIP......")
            
            
            
            i = i + 1
    
    # Histogram
    df_SSIM2.plot(kind = 'hist', legend = False, bins = 50, title = "Structural Similarity Index 2")
    
    
    # Get minimum values of everyrow
    
    df_SSIM2_Test = df_SSIM2
    indexes = df_SSIM2_Test.index
    
    
    df_SSIM2_Test['id_variable'] = indexes
    long_SSIM2 = pd.melt(df_SSIM2_Test, id_vars = ['id_variable'])
    
    long_SSIM2.dropna(subset = ['value'], inplace=True)
    long_SSIM2 = long_SSIM2[long_SSIM2.id_variable != long_SSIM2.variable]
    long_SSIM2 = long_SSIM2.sort_values(by = 'value')
    
    
    # Determine the File with the Poor Scores
    
    ImagesSSIM = long_SSIM2[['id_variable', 'value']]
    ImagesSSIM = ImagesSSIM.set_axis(["Image","SSIM_Score"], axis = 1)
    variableSSIM = long_SSIM2[['variable', 'value']]
    variableSSIM = variableSSIM.set_axis(["Image","SSIM_Score"], axis = 1)
    ImagesSSIM = ImagesSSIM.append(variableSSIM)
    
    # using dictionary to convert specific columns
    convert_dict = {'Image': str,
                    'SSIM_Score': float
                   }
      
    variableSSIM = variableSSIM.astype(convert_dict)
    
    
    ImageSSIM_Score = variableSSIM.groupby('Image').mean()
    #ImageSSIM_Score = ImageSSIM_Score.sort_values(by = 'SSIM_Score')
    
    
    # Histogram of SSIM Scores
    #ImageSSIM_Score.plot(kind = 'hist', legend = False, bins = 50, title = "Mean SSIM By Cluster Image")

    #Save Results
    clusterDictionary = {"CurrDirTail": CurrDirTail, "ImageSSIM_Score": ImageSSIM_Score, "Images": imagesCollected}    


    resultDictionary[CurrDirTail] = clusterDictionary



# MEan of cluster
meanImagesCollected = pd.DataFrame(np.concatenate(imagesCollected))

meanImagesCollected = pd.DataFrame(list(map(np.ravel, imagesCollected)))

meanImagesCollected = meanImagesCollected.mean(axis=0)

meanImagesCollected = meanImagesCollected.values.reshape((125,250))

plt.imshow(meanImagesCollected, interpolation='nearest')
plt.show()
    
plt.imshow(data_img, interpolation='nearest')
plt.show()
    
resultDictionary


for key in resultDictionary:
    print(key)

    dictOfImages = resultDictionary[key]['Images']

    # MEan of cluster
    #meanImagesColl = pd.DataFrame(np.concatenate(resultDictionary[key]['Images']))
    
    meanImagesColl = pd.DataFrame(list(map(np.ravel, dictOfImages)))
    
    meanImagesColl = meanImagesColl.mean(axis=0)
    
    meanImagesColl = meanImagesColl.values.reshape((125,250))
    
    # plt.imshow(meanImagesColl, interpolation='nearest')
    # plt.title(key,  fontweight ="bold")
    # plt.show()
    
    
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Output: " + key)
    ax1.imshow(meanImagesColl, interpolation='nearest')
    
    
    exampleGT_Image = resultDictionary[key]['Images'][10]
    exampleGT_Image = exampleGT_Image.reshape((125,250))
    
    ax2.imshow(exampleGT_Image, interpolation='nearest')
    
    histTitle = "Mean SSIM By Cluster Image: " + key
    
    resultDictionary[key]['ImageSSIM_Score'].plot(kind = 'hist', legend = False, bins = 50, title = histTitle)
    
    
    
    
    plt.imshow(data_img, interpolation='nearest')
    plt.show()
    






# Save Dictionary to Disk


import pickle

file_to_write = open("f:\MouthData.pickle", "wb")
pickle.dump(resultDictionary, file_to_write)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# ---------------------------------------------------------------------------------------------------------------------------



 
# END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END END
# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------


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