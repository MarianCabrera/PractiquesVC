# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:05:57 2021

@author: Pipo
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.signal

#Read all pixels
img = cv2.imread("s2.jpg")
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
plt.imshow(im_rgb)
count = 0
d = 3
k = 75
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
h = np.array([[0,0,0,1,0,0,0], 
              [0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0], 
              [1,0,0,0,0,0,1], 
              [0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0], 
              [0,0,0,1,0,0,0]], float)
red = scipy.signal.convolve2d(im_rgb[:,:,0],h, mode='same')/4
green = scipy.signal.convolve2d(im_rgb[:,:,1],h, mode='same')/4
blue = scipy.signal.convolve2d(im_rgb[:,:,2],h, mode='same')/4
# conv = scipy.signal.convolve2d(im_rgb,h, mode='same')/4

dist = np.sqrt((red[:,:]-im_rgb[:,:,0])**2 + (green[:,:]-im_rgb[:,:,1])**2 + (blue[:,:]-im_rgb[:,:,2])**2)   
size = d*2 + 1
m1 = np.ones((size, size))
for i in range(0,y-d):
    for j in range(0,x-d):
#         #print(im_rgb[i-d : i+d+1, j-d : j+d+1, 0])
#         r = np.convolve(im_rgb[:,:,0], [])

        # r = (im_rgb[i + d, j, 0] + im_rgb[i - d, j, 0]  + im_rgb[i, j + d, 0]  + im_rgb[i, j - d, 0])/4
        # g = (im_rgb[i + d, j, 1] + im_rgb[i - d, j, 1]  + im_rgb[i, j + d, 1]  + im_rgb[i, j - d, 1])/4
        # b = (im_rgb[i + d, j, 2] + im_rgb[i - d, j, 2]  + im_rgb[i, j + d, 2]  + im_rgb[i, j - d, 2])/4

        # op = np.sqrt((r-im_rgb[i,j,0] )**2+(g-im_rgb[i,j,1])**2+(b-im_rgb[i,j,2])**2)
        # op = np.sqrt((red[i,j]-im_rgb[i,j,0] )**2+(green[i,j]-im_rgb[i,j,1])**2+(blue[i,j]-im_rgb[i,j,2])**2)
        if dist[i,j] > k:
            newImg[i,j,0] = 255
            newImg[i,j,1] = 0
            newImg[i,j,2] = 0
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds') 
#Get all pixel's colors by freq rad
plt.imshow(newImg)
# print(im_rgb.shape)