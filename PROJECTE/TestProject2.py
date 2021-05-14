# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:05:57 2021

@author: Pipo
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

#Read all pixels
img = cv2.imread("samarreta.jpg")
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(im_rgb)
count = 0
d = 3
k = 350
y = im_rgb.shape[0]
x = im_rgb.shape[1]
newImg = np.zeros(im_rgb.shape)

# i = 10;
# j = 10;
# print(im_rgb[i, j, 0])
# print(im_rgb[i-1 : i+2, j-1 : j+2, 0])
# r = np.sum(im_rgb[i-1 : i+2, j-1 : j+2, 0])/9
#print(np.sqrt((255 )^2+(255 )^2+(255)^2))
t=time.time()
  
size = d*2 + 1
m1 = np.ones((size, size))
for i in range(d,y-d):
    for j in range(d,x-d):
#         #print(im_rgb[i-d : i+d+1, j-d : j+d+1, 0])

        r = (im_rgb[i + d, j, 0] + im_rgb[i - d, j, 0]  + im_rgb[i, j + d, 0]  + im_rgb[i, j - d, 0])/4
        g = (im_rgb[i + d, j, 1] + im_rgb[i - d, j, 1]  + im_rgb[i, j + d, 1]  + im_rgb[i, j - d, 1])/4
        b = (im_rgb[i + d, j, 2] + im_rgb[i - d, j, 2]  + im_rgb[i, j + d, 2]  + im_rgb[i, j - d, 2])/4

        op = np.sqrt((r-im_rgb[i,j,0] )**2+(g-im_rgb[i,j,1])**2+(b-im_rgb[i,j,2])**2)
        if op > k:
            newImg[i,j,0] = 255
            newImg[i,j,1] = 0
            newImg[i,j,2] = 0
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds') 
#Get all pixel's colors by freq rad
plt.imshow(newImg)
# print(im_rgb.shape)