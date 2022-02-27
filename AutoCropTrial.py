# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:52:22 2021

@author: WesSa
"""

    
imageToCrop = "f:/Tom_Hanks_face - Copy.jpg"    

from autocrop import Cropper
from PIL import Image

cropper = Cropper(face_percent = 80)
cropper = Cropper()

    
# Get a Numpy array of the cropped image
cropped_array = cropper.crop(imageToCrop)

# Save the cropped image with PIL if a face was detected:
#if cropped_array:
cropped_image = Image.fromarray(cropped_array)
cropped_image.save('f:/cropped_4.png')
