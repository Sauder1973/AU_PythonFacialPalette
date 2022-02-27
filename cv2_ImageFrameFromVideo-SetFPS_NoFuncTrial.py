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










Path = "F:"
TestVideo = "01-01-03-02-02-01-01.mp4" 
TestVideo = "02-01-05-01-01-01-12 - Angry Jane.mp4" 
fileName = Path +"/" + TestVideo

vidcap = cv2.VideoCapture(fileName) 


contrastLevel = 1.5
fps = 60  #(Higher the Value - The Lower the Frame Rate)
frameRate = fps/360  ##it will capture image in each 0.5 second 
sec = 0 

success = True

# Load the cascade
harrcascadePath = "C:/Users/WesSa/Python37_VENV/pyfeat37\Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
#face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(harrcascadePath)


facesMade = 0

while success: 
    facesMade = facesMade + 1
    sec = sec + frameRate 
    sec = round(sec, 2) 
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
    hasFrames,image = vidcap.read() 
    
    if hasFrames: 
        print('hasFrames')
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        #  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
          
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        
        
        # Draw rectangle around the faces and crop the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            # additionalOffsetDown = 30
            # additionalOffsetUp = 100
            # additionalOffsetLeft = 20
            # additionalOffsetRight = 10
            
            
            additionalOffsetDown = 0
            additionalOffsetUp = 0
            additionalOffsetLeft = 0
            additionalOffsetRight = 0
            
            

            
            
            faces = image[y - (additionalOffsetUp):y + (h + additionalOffsetDown), x - additionalOffsetLeft:x + (w + additionalOffsetRight)]

            # Change the contrast of the image:
            faces = cv2.addWeighted(faces, contrastLevel, np.zeros(faces.shape, faces.dtype), 0, 0)


            # Resizing the image provides better quality after face detect and Cropping
            faces = image_resize(faces,height= 1000)


            # Plott the CROPPER Face
            plt.imshow(faces, interpolation='nearest')
            plt.show()


            numpyFace = asarray(faces)
            
            
            cropper = Cropper(face_percent = 2)
            cropper = Cropper()

                
            # Get a Numpy array of the cropped image
            cropped_array = cropper.crop(numpyFace)

            # Plott the CROPPER Face
            plt.imshow(cropped_array, interpolation='nearest')
            plt.show()
            
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
            
            detected_faces = detector.detect_faces(numpyFace)
            
            
            
            detected_landmarks = detector.detect_landmarks(numpyFace, detected_faces)
            
            
            emotions = detector.detect_emotions(numpyFace, detected_faces, detected_landmarks)
            hogPyFeat = detector.extract_hog(numpyFace, orientation=8, 
                                              pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)



            # Action Unit Derivation --- WORKS WORKS WORKS WORKS !!!!                
            processedFramesPyFeat = detector.process_frame(numpyFace)
            
          
            
         
            
            
            
            
            # Add data, AU is ordered as such: 
            # AU1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 28, 43
            
            # Activate AU1: Inner brow raiser 
            # au = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # au = processedFramesPyFeat[['AU01','AU02','AU04','AU05','AU06','AU07','AU09',
            #                             'AU10','AU11','AU12','AU14','AU15','AU17','AU20',
            #                             'AU23','AU24','AU25','AU26','AU28','AU43']]
            
            Face_AU = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('AU')]
            Face_Coord_X = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('x_')]
            Face_Coord_Y = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('y_')]
            Face_DimDetails = processedFramesPyFeat.loc[:, processedFramesPyFeat.columns.str.startswith('Face')]
            Face_Emotion = processedFramesPyFeat[['anger', 'disgust','fear','happiness','sadness','surprise','neutral']]
            
            #Plot XY to determine what it has captured:


            tempX = Face_Coord_X.transpose()
            tempY = Face_Coord_Y.transpose()
                
            
            tempX.reset_index(drop=True, inplace=True)
            tempY.reset_index(drop=True, inplace=True)
            
                
            #faceONE = {'X': tempX[:][0], 'Y':tempY[:][0] }
            tempAU = Face_AU.transpose()
            
            #list(tempAU.index)
            
            tempAU.index.name = 'Action Unit'
            tempAU.reset_index(inplace=True)
            
            tempAU = pd.concat(tempAU, axis=1, keys=None)
            
            
            tempAU[[1][0:1]]
            
            tempAU = tempAU[[][0:1]]
            tempAU.columns = ["Action Unit","Value"]
            
            
            plt.barh(tempAU.index,tempAU[[1]], 
                     align='center', label="Data 1")
            
            
            
            
            
            
            
            
            
            
            
            faceONE = [tempX[0], tempY[0] ]
            faceONE = pd.concat(faceONE, axis=1, keys=None)
            faceONE.columns = ["X","Y"]
            faceONE["Y"] = -faceONE["Y"]
            faceONE["X"] = -faceONE["X"]
            
            
            # faceONE.plot(x='X', y='Y', style='o')
            


            faceAU_ONE = [tempX[0], tempY[0] ]
            faceAU_ONE = pd.concat(faceAU_ONE, axis=1, keys=None)
            faceAU_ONE.columns = ["X","Y"]
            faceAU_ONE["Y"] = -faceAU_ONE["Y"]
            faceAU_ONE["X"] = -faceAU_ONE["X"]

            
            faceTWO = [tempX[1], tempY[1] ]
            faceTWO = pd.concat(faceTWO, axis=1, keys=None)
            faceTWO.columns = ["X","Y"]
            
            # faceTWO.plot(x='X', y='Y', style='o')
            
            
            faceTHREE = [tempX[1], tempY[1] ]
            faceTHREE = pd.concat(faceTHREE, axis=1, keys=None)
            faceTHREE.columns = ["X","Y"]
            
            # faceTHREE.plot(x='X', y='Y', style='o')
            
            
            fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, figsize=(16, 32), sharex=False, sharey=False)
            ax1.axis('off')
            ax1.imshow(cropped_array, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            ax2.axis('off')
            ax2.imshow(numpyFace, cmap=plt.cm.gray)
            ax1.set_title('Contrast image')
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            ax3.axis('off')
            ax3.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax3.set_title('Histogram of Oriented Gradients')
            
            #ax4.axis('off')
            ax4.set_aspect(1/ax4.get_data_ratio(), adjustable = 'box')
            ax4.scatter(faceONE["X"],faceONE["Y"])
            ax4.set_title('Landmarks')
            
            
            
            plt.rcParams['figure.dpi'] = dpi
            plt.show()
            

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

















