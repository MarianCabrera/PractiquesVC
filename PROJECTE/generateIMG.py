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
s1 =  cv2.imread("samarretes/s1.png", -1)
s2 =  cv2.imread("samarretes/s2.png", -1)
s3 =  cv2.imread("samarretes/s3.png", -1)
s4 =  cv2.imread("samarretes/s4.png", -1)
s5 =  cv2.imread("samarretes/s5.png", -1)
sg =  cv2.imread("samarretes/sg.png", -1)

sizeX = im_rgb.shape[1]
sizeY = im_rgb.shape[0]
nRows = 40
mCols = 40
x = int(sizeX/mCols)
y = int(sizeY/nRows)

alpha_s = s1[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s

print(im_rgb.shape)

xS = s1.shape[1]
yS = s1.shape[0]

nS = 2
maxS = 3
minS = 1
sColor = 1
# s = s1

for i in range(0,nRows):
    for j in range(0, mCols):
        roi = img[i*y:i*y + y ,j*x:j*x + x]
        roi2 = img[i*y:i*y + y ,j*x:j*x + x]
        gt = np.zeros(roi.shape)
        nS = random.randint(minS,maxS)
            
        for k in range(nS):
            sColor = random.randint(1,5)
            rx = random.randint(xS,x-xS)
            ry = random.randint(yS,y-yS)
            if sColor == 1:
                roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s1[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
                roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s1[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
                roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s1[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
            elif sColor == 2: 
                roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s2[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
                roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s2[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
                roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s2[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
            elif sColor == 3:
                roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s3[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
                roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s3[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
                roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s3[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
            elif sColor == 4:
                roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s4[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
                roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s4[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
                roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s4[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
            else:
                roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s5[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
                roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s5[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
                roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s5[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
                
            
            # roi[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * s[:, :, 0] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 0])
            # roi[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * s[:, :, 1] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 1])
            # roi[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * s[:, :, 2] + alpha_l * roi[ry:ry+yS, rx:rx+xS, 2])
            gt[ry:ry+yS, rx:rx+xS, 0] = (alpha_s * sg[:, :, 0] + alpha_l * gt[ry:ry+yS, rx:rx+xS, 0])
            gt[ry:ry+yS, rx:rx+xS, 1] = (alpha_s * sg[:, :, 1] + alpha_l * gt[ry:ry+yS, rx:rx+xS, 1])
            gt[ry:ry+yS, rx:rx+xS, 2] = (alpha_s * sg[:, :, 2] + alpha_l * gt[ry:ry+yS, rx:rx+xS, 2])
        
        cv2.imwrite('raw/r_'+str(i)+'_'+str(j)+".jpg", roi2)
        cv2.imwrite('dataset/i_'+str(i)+'_'+str(j)+".jpg", roi)
        cv2.imwrite('groundtruth/g_'+str(i)+'_'+str(j)+".jpg", gt)
        
