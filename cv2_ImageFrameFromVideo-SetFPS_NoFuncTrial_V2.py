# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:22:13 2021

@author: WesSa


# This process works very good!!!!
# Use this process as required.

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



face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)




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



RyersonAV_Path = "F:/RyersonAV/RyersonActorFiles/ActorVideoOnly"
CurrentActor = "Actor_12"

RyersonAV_OutPath = "F:/ImageFrameFromVideo_TrialResults/Actor_12_Results"






Path = "F:"
TestVideo = "01-01-03-02-02-01-01.mp4" 
TestVideo = "02-01-05-01-01-01-12 - Angry Jane.mp4" 
fileName = Path +"/" + TestVideo


# # TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_16_10_37_Pro.mp4"
# # TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_16_16_45_Pro.mp4"
#TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_17_17_58_Pro.mp4"
#TestVideo = "C:/Users/WesSa/OneDrive/Pictures/Camera Roll/WIN_20211028_17_34_18_Pro.mp4"

#fileName = TestVideo

vidcap = cv2.VideoCapture(fileName) 


contrastLevel = 1.5
fps = 30  #(Higher the Value - The Lower the Frame Rate)
frameRate = fps/360  ##it will capture image in each 0.5 second 
sec = 0 

success = True

# Load the cascade
harrcascadePath = "C:/Users/WesSa/Python37_VENV/pyfeat37\Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
#face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(harrcascadePath)


facesMade = 0

while success: 
    currFile = fileName
    facesMade = facesMade + 1
    sec = sec + frameRate 
    sec = round(sec, 2) 
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
    hasFrames,OrigImage = vidcap.read() 
    print(hasFrames)

    if hasFrames: 
    
        #Preview Image:  - Colour is off since it is BGR NOT RGB
        # plt.axis("off")
        # plt.imshow(cv2.cvtColor(OrigImage, cv2.COLOR_BGR2RGB))
        # plt.show()
    
        # IMAGE is of TYPE:  numpy.ndarray    
        type(OrigImage)
        # Convert into grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray, interpolation='nearest')
        # Adjust Contrast
        image = OrigImage
        image = cv2.addWeighted(OrigImage, contrastLevel, np.zeros(OrigImage.shape, OrigImage.dtype), 0, 0)
        # Resizing the image provides better quality after face detect and Cropping
        image = image_resize(image,height= 1000)

        # Preview POST Contrast and Resizing
        # plt.axis("off")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()


        # Convert to Numpy Array  - NOT SURE THIS IS STILL NEEDED!!!
        # type(image)
        # numpyFace = asarray(image)
        # type(numpyFace)

 
        # Use Cropper to Detect and Crop Face
        cropper = Cropper(face_percent = 50)
        cropped_array = cropper.crop(image)
 

        croppedImageFound = type(cropped_array)
        
        if croppedImageFound is not type(None):
            # Plott the CROPPER Face
            # plt.imshow(cropped_array, interpolation='nearest')
            # plt.show()

            

        # Extract Histogram of Gradients - HOG        
            orientations = 8
            pixels_per_cell = (8,8)
            cells_per_block = (2,2)
            multiChannel = True
            dpi = 60
            #Generate HOG:
            
            fd, hog_image = hog(cropped_array, orientations = orientations,    
                    pixels_per_cell = pixels_per_cell,   
                    cells_per_block = cells_per_block,   
                    visualize = True,            
                    multichannel = multiChannel,         
                    feature_vector = True)   
            
                
               
                
            # PyFeat Pull Action Units
            # ALL WORKS WHEN USING IMAGE DATA - NOT THE FILEPATH  
                
            detected_faces = detector.detect_faces(cropped_array)
            detected_landmarks = detector.detect_landmarks(cropped_array, detected_faces)
            emotions = detector.detect_emotions(cropped_array, detected_faces, detected_landmarks)
            
            PyFeat_fd,PyFeat_hog_image = detector.extract_hog(cropped_array, orientation=8, 
                                      pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    
    
            # Action Unit Derivation --- WORKS WORKS WORKS WORKS !!!!                
            processedFramesPyFeat = detector.process_frame(cropped_array)
            
            
            # Pull the resulting dataframe apart to capture primary ingredients.
            Face_AU = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('AU')]
            Face_Coord_X = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('x_')]
            Face_Coord_Y = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('y_')]
            Face_DimDetails = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('Face')]
            Face_Emotion = processedFramesPyFeat[['anger', 'disgust','fear','happiness','sadness','surprise','neutral']]
    
            # Build Matrix for Landmark Data
             
            #Plot XY to determine what it has captured:
            
            
            tempX = Face_Coord_X.transpose()
            tempY = Face_Coord_Y.transpose()
            
            
            tempX.reset_index(drop=True, inplace=True)
            tempY.reset_index(drop=True, inplace=True)
            
            
            faceONE = [tempX, tempY ]
            faceONE = pd.concat(faceONE, axis=1, keys=None)
            faceONE.columns = ["X","Y"]
            faceONE["Y"] = -faceONE["Y"]
            faceONE["X"] = faceONE["X"]
    
            # Plot Results from Emotional Analysis
        
            Face_Emotion_Long = Face_Emotion
            Face_Emotion_Long['FaceID'] = facesMade
            
            Face_Emotion_Long = Face_Emotion_Long.set_index('FaceID').T
            Face_Emotion_Long['Emotion'] = Face_Emotion_Long.index
            Face_Emotion_Long.index = (str('00000000' + str(facesMade))[-7:] + "_" + Face_Emotion_Long['Emotion'] )
            Face_Emotion_Long['FaceID'] = facesMade
            Face_Emotion_Long = Face_Emotion_Long.set_axis(['Score','Emotion','Face_ID'], axis=1, inplace=False)
            
            Plot_emotion = Face_Emotion_Long['Emotion'].tolist()
            Plot_score = Face_Emotion_Long['Score'].tolist()
            # plt.barh(Plot_emotion,Plot_score)
            
            
            # Plot Results from Action Units
        
            Face_AU_Long = Face_AU
            Face_AU_Long['FaceID'] = facesMade
            
            Face_AU_Long = Face_AU_Long.set_index('FaceID').T
            Face_AU_Long['ActionUnit'] = Face_AU_Long.index
            Face_AU_Long.index = (str('00000000' + str(facesMade))[-7:] + "_" + Face_AU_Long['ActionUnit'] )
            Face_AU_Long['FaceID'] = facesMade
            Face_AU_Long = Face_AU_Long.set_axis(['Score','ActionUnit','Face_ID'], axis=1, inplace=False)
            
            Plot_ActionUnit = Face_AU_Long['ActionUnit'].tolist()
            AU_Plot_score = Face_AU_Long['Score'].tolist()
            #plt.barh(Plot_ActionUnit,AU_Plot_score)
        

    
    
    
            
            fig, (ax1, ax2,ax3, ax5, ax4,ax6) = plt.subplots(nrows = 1,ncols = 6, figsize=(16, 32), sharex=False, sharey=False)
            #fig, (ax4) = plt.subplots(nrows = 1,ncols = 1, figsize=(4, 8), sharex=False, sharey=False)
            ax1.axis('off')
            ax1.imshow(OrigImage, cmap=plt.cm.gray)
            ax1.set_title('Original Image')
            
            ax2.axis('off')
            ax2.imshow(cropped_array, cmap=plt.cm.gray)
            ax2.set_title('Cropped Image')
            
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            ax3.axis('off')
            ax3.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax3.set_title('Histogram of Oriented Gradients')

            # # Rescale histogram for better display
            # PyFeat_hog_image_rescaled = exposure.rescale_intensity(PyFeat_hog_image, in_range=(0, 10))
            # ax4.axis('off')
            # ax4.imshow(PyFeat_hog_image_rescaled, cmap=plt.cm.gray)
            # ax4.set_title('PyFeat - Histogram of Oriented Gradients')
          
            
            #aspectRatio = 1/ax3.get_data_ratio()
            aspectRatio = 1/8
            
            ax5.axis('off')
            ax5.set_aspect(1/ax3.get_data_ratio(), adjustable = 'box')
            ax5.scatter(faceONE["X"],faceONE["Y"])
            ax5.set_title('Landmarks')

            
            #ax4.axis('off')
            ax4.set_aspect(aspectRatio, adjustable = 'box')
            ax4.barh(Plot_emotion,Plot_score)
            ax4.set_title('Emotions')

            ax6.set_aspect(aspectRatio, adjustable = 'box')
            ax6.barh(Plot_ActionUnit,AU_Plot_score)
            ax6.set_title('Action Units')




            #ax4.barh(Plot_emotion,Plot_score, height = 1)
            
            
            plt.rcParams['figure.dpi'] = dpi
            plt.show()
            
            
            
            #Add Results to the Master Tables:
               
                
            # Landmark Table
            # Emotion Table
            # AU

        

        else:
            print("NO CROPPING :(")
            


    else:
        print("NoFrames")
        success = False
        cv2.destroyAllWindows()












#             # Save the cropped image with PIL if a face was detected:
#             #if cropped_array:
#             cropped_image = Image.fromarray(cropped_array)


#             cv2.imshow("face",cropped_image)
#             #cv2.waitKey()
#             cv2.waitKey(100)
            
            
            
        
            
            
            
#        #     cv2.imshow("face",faces)
#        #     #cv2.waitKey()
#        #     cv2.waitKey(100)
           
            
#             # Write to Disk
#             #cv2.imwrite('face.jpg', faces)
            
#     else:
#         print("NoFrames")
#         success = False
#         cv2.destroyAllWindows()
        
        
        
      



#         detector = Detector
#         image_prediction = detector.detect_image(image)
#         # Show results
#         image_prediction
#         # Show results
#         image_prediction
#         image_prediction.plot_detections()
        
#     #sec = sec + frameRate 
#     #sec = round(sec, 2)
#     #success = hasFrames
    

    
    
    
  
# print("END OF FRAME")    
        
    
# cv2.destroyAllWindows()
    
    
    
    
    
# vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
# hasFrames,image = vidcap.read() 
# if hasFrames:
#     #Show image:
#     cv2.imshow('image',image)
#     cv2.waitKey(0)
#     #image_prediction = detector.detect_image(image)
#     # Show results
#     #image_prediction
#     #image_prediction.plot_detections();
#     image_prediction = detector.detect_image(success)




















# while success: 
#     sec = sec + frameRate 
#     sec = round(sec, 2) 
#     success = getFrame(sec)
    
    
    
    
    
    
    
    
    
    
    
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
#     hasFrames,image = vidcap.read() 
#     if hasFrames:
#         #Show image:
#         cv2.imshow('image',image)
#         cv2.waitKey(0)
#         #image_prediction = detector.detect_image(image)
#         # Show results
#         #image_prediction
#         #image_prediction.plot_detections();
#         image_prediction = detector.detect_image(success)
    
    
    
    
    
    
#     #getFrame.hasFrames
    

#     # Show results
#     #image_prediction
#     success.plot_detections();
    
# cv2.destroyAllWindows()

















