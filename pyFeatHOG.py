# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:29:06 2022

@author: WesSa
"""

from PIL import Image, ImageOps
import math
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from feat import Detector
from feat.tests.utils import get_test_data_path
import os
import matplotlib.pyplot as plt
## EXTRACT HOGS
from skimage import data, exposure
from skimage.feature import hog
from tqdm import tqdm
import cv2
import pandas as pd
import csv
import numpy as np

def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """
    Adapted from https://github.com/ZhiwenShao/PyTorch-JAANet by Zhiwen Shao, modified by Tiankang Xie
    img: image
    img_land: landmarks 68
    box_enlarge: relative size of face
    img_size = 112
    """

    leftEye0 = (img_land[2 * 36] + img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] + img_land[2 * 40] +
                img_land[2 * 41]) / 6.0
    leftEye1 = (img_land[2 * 36 + 1] + img_land[2 * 37 + 1] + img_land[2 * 38 + 1] + img_land[2 * 39 + 1] +
                img_land[2 * 40 + 1] + img_land[2 * 41 + 1]) / 6.0
    rightEye0 = (img_land[2 * 42] + img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] + img_land[2 * 46] +
                 img_land[2 * 47]) / 6.0
    rightEye1 = (img_land[2 * 42 + 1] + img_land[2 * 43 + 1] + img_land[2 * 44 + 1] + img_land[2 * 45 + 1] +
                 img_land[2 * 46 + 1] + img_land[2 * 47 + 1]) / 6.0
    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 30], img_land[2 * 30 + 1], 1],
                   [img_land[2 * 48], img_land[2 * 48 + 1], 1], [img_land[2 * 54], img_land[2 * 54 + 1], 1]])
    mat2 = (mat1 * mat2.T).T
    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1
    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.array(list(zip(new_land[:,0], new_land[:,1]))).astype(int)
    return aligned_img, new_land


def extract_hog(image, detector):
    im = cv2.imread(image)
    detected_faces = np.array(detector.detect_faces(im)[0])
    im = np.asarray(im)
    detected_faces = np.array(detector.detect_faces(np.array(im))[0])
    detected_faces = detected_faces.astype(int)
    points = detector.detect_landmarks(np.array(im), [detected_faces])[0].astype(int)

    aligned_img, points = align_face_68pts(im, points.flatten(), 2.5)

    hull = ConvexHull(points)
    mask = grid_points_in_poly(shape=np.array(aligned_img).shape, 
                               verts= list(zip(points[hull.vertices][:,1], points[hull.vertices][:,0])) # for some reason verts need to be flipped
                               )
    mask[0:np.min([points[0][1], points[16][1]]), points[0][0]:points[16][0]] = True
    aligned_img[~mask] = 0
    resized_face_np = aligned_img

    fd, hog_image = hog(resized_face_np, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

    return fd, points.flatten(), resized_face_np, hog_image

