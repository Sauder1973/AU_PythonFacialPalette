# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:35:49 2022

@author: WesSa
"""

import torch
from torch import nn
from tqdm.auto import tqdm

import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision.transforms import Resize


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random

from PIL import Image
import os

import pandas as pd
import numpy as np



import glob
import cv2
from sys import argv

import sewar as simu
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from skimage.metrics import structural_similarity as ssim2



# Root directory for dataset

    

dirForPics = "F:/Project_FacialReenactmentOutputs/Actor_12/FacesOut_2022_02_11_HR_01_result_LeftEye/"
dirForPics = "F:/Project_FacialReenactmentOutputs/Actor_12/FacesOut_2022_02_11_HR_01_result_Mouth/"
#dataroot =          "f:/Project_FacialReenactmentOutputs/Actor12/FacesOut_2022_02_11_HR_01_result_LeftEye/"
#directoryOfImages = "f:/Project_FacialReenactmentOutputs/Actor12/DATA_ACTOR_12_SINGLE_DIRECTORY/FacesOut_2022_01_18_HR_03/"



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
    
    # df_MSE  = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_RMSE = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_PSNR = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_SSIM = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    
    df_SSIM2 = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    
    # df_UQI  = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_ERGA = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_SCC  = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_RASE = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_SAM  = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    # df_VIF  = pd.DataFrame(index=range(len(FileList)),columns=range(len(FileList)))
    
    
    # df_MSE.columns = FileList
    # df_MSE.index = FileList
    
    # df_RMSE.columns = FileList
    # df_RMSE.index = FileList
    
    # df_PSNR.columns = FileList
    # df_PSNR.index = FileList
    
    # df_SSIM.columns = FileList
    # df_SSIM.index = FileList
    
    df_SSIM2.columns = FileList
    df_SSIM2.index = FileList
    
    # df_UQI.columns = FileList
    # df_UQI.index = FileList
    
    # df_ERGA.columns = FileList
    # df_ERGA.index = FileList
    
    # df_SCC.columns = FileList
    # df_SCC.index = FileList
    
    # df_RASE.columns = FileList
    # df_RASE.index = FileList
    
    # df_SAM.columns = FileList
    # df_SAM.index = FileList
    
    # df_VIF.columns = FileList
    # df_VIF.index = FileList
    

    
    
    imagesCollected = []
    
    #Testfile = FileList[1]
    for Testfile in FileList:
    
        print("Cluster: " + CurrDirTail +"  Primary File: " + Testfile)
        test_path = os.path.join(data_dir, Testfile)
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        
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
                # df_MSE.loc [Testfile, Curfile] = simu.mse(test_img,data_img) 
                #df_RMSE.loc[Testfile, Curfile] = simu.rmse(test_img, data_img)
                #df_PSNR.loc[Testfile, Curfile] = simu.psnr(test_img, data_img)
                #df_SSIM.loc[Testfile, Curfile] = simu.ssim(test_img, data_img)
                
                df_SSIM2.loc[Testfile, Curfile] = ssim2(test_img, data_img, multichannel=True)
                
                
                #df_UQI.loc [Testfile, Curfile] = simu.uqi(test_img, data_img)
                #df_ERGA.loc[Testfile, Curfile] = simu.ergas(test_img, data_img)
                #df_SCC.loc [Testfile, Curfile] = simu.scc(test_img, data_img)
                #df_RASE.loc[Testfile, Curfile] = simu.rase(test_img, data_img)
                #df_SAM.loc [Testfile, Curfile] = simu.sam(test_img, data_img)
                #df_VIF.loc [Testfile, Curfile] = simu.vifp(test_img, data_img)
            else:
                print("Test and Current Image are the same SKIP......")
            
            
            
            i = i + 1
    
    
    
    
    
    
    # Histogram
    
    
    df_SSIM2.plot(kind = 'hist', legend = False, bins = 50, title = "Structural Similarity Index 2")
    
    #df_MSE.plot(kind = 'hist', legend = False, bins = 50, title = "Mean Square Error")
    #df_RMSE.plot(kind = 'hist', legend = False, bins = 50, title = "Root Mean Square Error")
    #df_PSNR.plot(kind = 'hist', legend = False, bins = 30, title = "Peak Signal to Noise Ratio")
    #df_SSIM.plot(kind = 'hist', legend = False, bins = 30, title = "Structural Similarity Index")
    #df_UQI.plot(kind = 'hist', legend = False, bins = 30, title = "Universal Quality Image Index")
    #df_ERGA.plot(kind = 'hist', legend = False, bins = 30)
    #df_SCC.plot(kind = 'hist', legend = False, bins = 30)
    #df_RASE.plot(kind = 'hist', legend = False, bins = 30)
    #df_SAM.plot(kind = 'hist', legend = False, bins = 30)
    #df_VIF.plot(kind = 'hist', legend = False, bins = 30)
    
    
    
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












# Loop through each pair - sanity check:

for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    
    plt.imshow(data_img, interpolation='nearest')
    plt.show()
    

#
# ----------------------------------------------------------------------------------------------------
# Multiple Images Trial 


MSE  = {}
RMSE = {}
PSNR = {}
SSIM = {}
UQI  = {}
ERGA = {}
SCC  = {}
RASE = {}
SAM  = {}
VIF  = {}


data_dir = dirForPics + "CLUSTER0/"
FileList = os.listdir(data_dir)

test_path = data_dir + data_dir[1]

test_img = cv2.imread(test_path)

i = 0

# test implementation:
    
for Testfile in FileList:

    print("Primary File: " + Testfile)
    test_path = os.path.join(data_dir, Testfile)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
   # test_img = cv2.threshold(test_img, args["threshold"], 255, cv2.THRESH_TRUNC)
    
    #plt.imshow(test_img, interpolation='nearest')
    #plt.show()

    for Curfile in FileList[i:len(FileList)]:
        #if img_path != test_path: 
        print("Current File: " + str(i) + " Filename:" +Curfile)
        img_path = os.path.join(data_dir, Curfile)
        data_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        
    i = i + 1
                





# ----------------------------------------------------------------------------------------------------
# Single Image Trial 


data_dir = dirForPics + "CLUSTER0/"
img = "Actor_12_Image_0599_result_LeftEye.jpg"
test_path = data_dir + img

test_img = cv2.imread(test_path)




MSE  = {}
RMSE = {}
PSNR = {}
SSIM = {}
SSIM2 = {}
UQI  = {}
ERGA = {}
SCC  = {}
RASE = {}
SAM  = {}
VIF  = {}


plt.imshow(test_img, interpolation='nearest')
plt.show()



for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    
    plt.imshow(data_img, interpolation='nearest')
    plt.show()
    
    if img_path != test_path:
    
       # MSE[img_path]  = simu.mse(test_img,data_img)
       # RMSE[img_path] = simu.rmse(test_img, data_img)
       # PSNR[img_path] = simu.psnr(test_img, data_img)
       # SSIM[img_path] = simu.ssim(test_img, data_img)
       
        SSIM2[img_path] = ssim2(test_img, data_img, multichannel=True)
        
       # UQI[img_path]  = simu.uqi(test_img, data_img)
       # ERGA[img_path] = simu.ergas(test_img, data_img)
       # SCC[img_path]  = simu.scc(test_img, data_img)
       # RASE[img_path] = simu.rase(test_img, data_img)
       # SAM[img_path]  = simu.sam(test_img, data_img)
       # VIF[img_path]  = simu.vifp(test_img, data_img)
    else:
        print("Test and Current Image are the same SKIP......")
        



df_MSE  = pd.DataFrame.from_dict(MSE , orient = 'index')
df_RMSE = pd.DataFrame.from_dict(RMSE, orient = 'index')
df_PSNR = pd.DataFrame.from_dict(PSNR, orient = 'index')
df_SSIM = pd.DataFrame.from_dict(SSIM, orient = 'index')
df_SSIM2 = pd.DataFrame.from_dict(SSIM2, orient = 'index')

df_UQI  = pd.DataFrame.from_dict(UQI , orient = 'index')
df_ERGA = pd.DataFrame.from_dict(ERGA, orient = 'index')
df_SCC  = pd.DataFrame.from_dict(SCC , orient = 'index')
df_RASE = pd.DataFrame.from_dict(RASE, orient = 'index')
df_SAM  = pd.DataFrame.from_dict(SAM , orient = 'index')
df_VIF  = pd.DataFrame.from_dict(VIF , orient = 'index')
        
        

def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
        closest = max(dict.values())
    else:
        closest = min(dict.values())

    for key, value in dict.items():
        print("The difference between ", key ," and the original image is : \n", value)
        if (value == closest):
            result[key] = closest

    print("The closest value: ", closest)
    print("######################################################################")
    return result



ssim = calc_closest_val(SSIM, True)
rmse = calc_closest_val(RMSE, False)
sre = calc_closest_val(UQI, True)

print("The most similar according to SSIM: " , ssim)
print("The most similar according to RMSE: " , rmse)
print("The most similar according to SRE: " , sre)

        


