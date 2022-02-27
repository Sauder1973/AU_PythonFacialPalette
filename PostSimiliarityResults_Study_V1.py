# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:24:44 2022

@author: WesSa
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fileToUnPack = "f:/LeftEyeData.pickle"
fileToUnPack = "f:/MouthData.pickle"

import pickle


with open(fileToUnPack, 'rb') as f:
    resultDictionary = pickle.load(f)

for key in resultDictionary:
    print(key)

    dictOfImages = resultDictionary[key]['Images']
    meanImagesColl = pd.DataFrame(list(map(np.ravel, dictOfImages)))
    meanImagesColl = meanImagesColl.mean(axis=0)
    meanImagesColl = meanImagesColl.values.reshape((125,250))
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('OUTPUT:' + key)
    ax1.imshow(meanImagesColl, interpolation='nearest')
    
    
    exampleGT_Image = resultDictionary[key]['Images'][10]
    exampleGT_Image = exampleGT_Image.reshape((125,250))
    
    ax2.imshow(exampleGT_Image, interpolation='nearest')
    
    histTitle = "Mean SSIM By Cluster Image: " + key
    
    resultDictionary[key]['ImageSSIM_Score'].plot(kind = 'hist', legend = False, bins = 50, title = histTitle)
    
    


outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

for key in resultDictionary:
    print(key)
    dataset = resultDictionary[key]['ImageSSIM_Score']
    dataset = dataset.values.tolist()

    outlier_datapoints = detect_outlier(dataset)
    print(outlier_datapoints)
    myList = list(np.around(np.array(outlier_datapoints),1))
    plt.hist(myList, 50)
    plt.show()






