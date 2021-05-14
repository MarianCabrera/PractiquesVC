# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:17:40 2021

@author: Pipo
"""
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from PIL import Image

img = cv2.imread("gps.jpg")
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
s =  cv2.imread("samarretes/s1.png", -1)
sg =  cv2.imread("samarretes/sg.png", -1)

sizeX = im_rgb.shape[1]
sizeY = im_rgb.shape[0]
nRows = 20
mCols = 20

alpha_s = s[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s

print(im_rgb.shape)

xS = s.shape[1]
yS = s.shape[0]

for i in range(0,nRows):
    for j in range(0, mCols):
        x = int(sizeX/mCols)
        y = int(sizeY/nRows)
        roi = img[i*y:i*y + y ,j*x:j*x + x]
        rx = random.randint(10,x-10)
        ry = random.randint(10,y-10)
        
        roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
        roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
        roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
        
        cv2.imwrite('dataset/i_'+str(i)+'_'+str(j)+".jpg", roi)
        
        gt = np.zeros(roi.shape)
        gt[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * sg[:, :, 0] + alpha_l * gt[ry:ry+yS, rx:rx+xS, 0])
        gt[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * sg[:, :, 1] + alpha_l * gt[ry:ry+yS, rx:rx+xS, 1])
        gt[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * sg[:, :, 2] + alpha_l * gt[ry:ry+yS, rx:rx+xS, 2])
                
        cv2.imwrite('groundtruth/g_'+str(i)+'_'+str(j)+".jpg", gt)
        
