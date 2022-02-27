# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:27:38 2021

@author: WesSa

@title: 

"""

Path = "F:"
TestVideo = "01-01-03-02-02-01-01.mp4"
fileName = Path +"/" + TestVideo





# Find the file you want to process.
from feat.tests.utils import get_test_data_path
import os, glob

test_video = os.path.join(Path,TestVideo)

#test_data_dir = get_test_data_path()
#test_video = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

# Show video
from IPython.display import Video
Video(test_video, embed=True)


from feat import Detector
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)


video_prediction = detector.detect_video(test_video, skip_frames=24)
video_prediction.head(10)

video_prediction



video_prediction.loc[[72]].plot_detections();


















# Find the file you want to process.
#from feat.tests.utils import get_test_data_path
import os, glob
test_data_dir = get_test_data_path()

test_video = os.path.join(Path,TestVideo)


# Show video
from IPython.display import Video
Video(test_video, embed=True)


video_prediction = detector.detect_video(test_video, skip_frames=24)
video_prediction.head()


video_prediction.loc[[12]].plot_detections();

video_prediction.emotions().plot()

video_prediction = detector.detect_video(test_video)
video_prediction