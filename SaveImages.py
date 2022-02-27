# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 23:27:50 2021

SAVE Data - By Cluster and Mass Save to Directory


@author: WesSa
"""

import pandas as pd
import os
import datetime
import pytz
from pathlib import Path
from PIL import Image


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



def Save_FileToClusterDirctory(out_Labels, result_Images, IDString, Root,FacePart ):
        
    indexes = list(range(0,len(out_Labels)))
    
    outClusters = pd.DataFrame({
                        'Index':   indexes,
                        'Cluster': out_Labels,
                        'Image':   result_Images})
    
    my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
    studyPath = Root
    
    newDir = "FacesOut_" + str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:])
    
    DirToCreate = studyPath + IDString + "/" + newDir + "/"  
    Path(DirToCreate).mkdir(parents=True, exist_ok=True)
    
    outClusters["FileName"] = outClusters.apply(lambda row: IDString + "_Image_" + ("0000" + str(row.Index))[-4:]+"_"+FacePart+ ".jpg", axis = 1)

# Save By Cluster Directory ------------------------------------------------------------------------------------------------------



    # Save each cluster to its own directory --------------------------------------------------------------------
    clusterCounts =  outClusters.Cluster.value_counts()
    clusterCounts = clusterCounts.to_frame()
    clusterCounts['blank'] = clusterCounts.index
    clusterCounts.set_axis(['Counts','Cluster'], axis = 1, inplace = True)
    clusterCounts["ClusterDirectory"] = clusterCounts.apply(lambda row: IDString + "_Cluster_" + ("0000" + str(row.Cluster))[-4:], axis = 1)
    
    
    newDir = "FacesOut_" + str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:]+"_"+FacePart)
    
    
    
    
    for currCluster in range(0,len(clusterCounts)):
        
        print(currCluster)
        print(clusterCounts.ClusterDirectory[currCluster])
        
        DirToCreate = studyPath + IDString + "/" + newDir + "/CLUSTER" + str(currCluster) + "/"
        # Path(DirToCreate).mkdir(parents=True, exist_ok=True)
        ensure_dir(DirToCreate)
    
        ImagesFromCluster   = outClusters.loc[outClusters['Cluster']==currCluster,'Image']
        FileNameFromCluster = outClusters.loc[outClusters['Cluster']==currCluster,'FileName']
        
        ImageIndex = ImagesFromCluster.index
        
        for tmpImageIndex in ImageIndex:
            tmpImage    = ImagesFromCluster[tmpImageIndex]
            tmpFileName = FileNameFromCluster[tmpImageIndex]
    
            tmpFileName = DirToCreate + "/" + tmpFileName       
    
            im = Image.fromarray(tmpImage)
            im.save(tmpFileName)
        




def Save_FilesToSingleDirctory(out_Labels, result_Images, IDString, Root, FacePart, SaveCSV = False ):
        
    listOfFileNames = []
    listOfClusters = []
    
    indexes = list(range(0,len(out_Labels)))
    
    outClusters = pd.DataFrame({
                        'Index':   indexes,
                        'Cluster': out_Labels,
                        'Image':   result_Images})
    
    my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
    studyPath = Root
    
    newDir = "FacesOut_" + str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:]+"_"+FacePart)
    
    DirToCreate = studyPath + IDString + "/" + newDir + "/"  
    Path(DirToCreate).mkdir(parents=True, exist_ok=True)
    
    outClusters["FileName"] = outClusters.apply(lambda row: IDString + "_Image_" + ("0000" + str(row.Index))[-4:]+"_"+FacePart+ ".jpg", axis = 1)

# Save By Cluster Directory ------------------------------------------------------------------------------------------------------



    # Save each cluster to its own directory --------------------------------------------------------------------
    clusterCounts =  outClusters.Cluster.value_counts()
    clusterCounts = clusterCounts.to_frame()
    clusterCounts['blank'] = clusterCounts.index
    clusterCounts.set_axis(['Counts','Cluster'], axis = 1, inplace = True)
    clusterCounts["ClusterDirectory"] = clusterCounts.apply(lambda row: IDString + "_Cluster_" + ("0000" + str(row.Cluster))[-4:], axis = 1)
    
    
    newDir = "FacesOut_" + str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:])
    
    
    
    
    for currCluster in range(0,len(clusterCounts)):
        
        print(currCluster)
        print(clusterCounts.ClusterDirectory[currCluster])
        
        # DirToCreate = studyPath + IDString + "/" + newDir + "/CLUSTER" + str(currCluster) + "/"
        # ensure_dir(DirToCreate)
    
        ImagesFromCluster   = outClusters.loc[outClusters['Cluster']==currCluster,'Image']
        FileNameFromCluster = outClusters.loc[outClusters['Cluster']==currCluster,'FileName']
        
        ImageIndex = ImagesFromCluster.index
        
        for tmpImageIndex in ImageIndex:
            tmpImage    = ImagesFromCluster[tmpImageIndex]
            tmpFileName = FileNameFromCluster[tmpImageIndex]
    
            # tmpFileName = DirToCreate + "/" + tmpFileName       
            tmpFileName = DirToCreate + "/" + tmpFileName       
    
            im = Image.fromarray(tmpImage)
            im.save(tmpFileName)
            
            listOfFileNames.append(tmpFileName)
            listOfClusters.append(currCluster)
            
            
    # if SaveCSV == True:
    #     # Clean up File Name
    #     fileNameIterator = map(lambda fileName: os.path.basename(fileName), listOfFiles)

    #     listOfFilesNames = (list(map(lambda fileName: os.path.basename(fileName), listOfFiles)))
    #     # listOfFiles = [os.path.splitext(x)[0] for x in listOfFilesNames]


    #     #Build CSV - FinalPath, File Name, Cluster

    #     list_of_Tuples = list(zip(listOfFiles, listOfFilesNames, listOfClusters))
    #     df_ImageToCluster = pd.DataFrame(list_of_Tuples,columns = ['FileAndPath', 'Filename', 'Cluster'])

    #     df_ImageToCluster.to_csv(directoryOfImages+"df_ImageToCluster.csv", index = False)
        
    return listOfFileNames, listOfClusters, DirToCreate


def Save_FilesToSingleDir_NoCluster(result_Images, Root, FacePart, SaveCSV = False ):
      
    listOfFileNames = []
    
    indexes = list(range(0,len(result_Images)))
    
    outImages = pd.DataFrame({
                        'Index':   indexes,
                        'Image':   result_Images})
    
    my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
   
    
    newDir = FacePart + "_" +  str(my_date.year) + "_" + str(("00" + str(my_date.month))[-2:]) + "_" + str(("00" + str(my_date.day))[-2:]) + "_HR_" +  str(("00" + str(my_date.hour))[-2:] +  str(("00" + str(my_date.minute))[-2:]) )
    DirToCreate = Root + "/" + newDir + "/"  
    #Path(DirToCreate).mkdir(parents=True, exist_ok=True)
    os.mkdir(DirToCreate)

    outImages["FileName"] = outImages.apply(lambda row: "Image_" + ("0000" + str(row.Index))[-4:]+"_"+FacePart+ ".jpg", axis = 1)

# Save Each File to Directory ------------------------------------------------------------------------------------------------------

    tmpImageIndex = 0

    while tmpImageIndex < len(result_Images):
        tmpImage    = result_Images[tmpImageIndex]
        tmpFileName = outImages['FileName'][tmpImageIndex]

        # tmpFileName = DirToCreate + "/" + tmpFileName       
        tmpFileName = DirToCreate + "/" + tmpFileName       

        im = Image.fromarray(tmpImage)
        im.save(tmpFileName)
        
        listOfFileNames.append(tmpFileName)
        tmpImageIndex += 1
        
    return listOfFileNames, DirToCreate