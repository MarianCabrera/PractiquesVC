# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:17:13 2021

@author: Pipo
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.signal

def bestoSercho(im, d, k):
    result = np.zeros(im.shape)
    
    for i in range(d,im.shape[0]-d):
        for j in range(d,im.shape[1]-d):
            r = (im[i + d, j, 0] + im[i - d, j, 0]  + im[i, j + d, 0]  + im[i, j - d, 0])/4
            g = (im[i + d, j, 1] + im[i - d, j, 1]  + im[i, j + d, 1]  + im[i, j - d, 1])/4
            b = (im[i + d, j, 2] + im[i - d, j, 2]  + im[i, j + d, 2]  + im[i, j - d, 2])/4

            op = np.sqrt((r-im[i,j,0] )**2+(g-im[i,j,1])**2+(b-im[i,j,2])**2)
            if op > k:
                result[i,j,0] = 1
                result[i,j,1] = 1
                result[i,j,2] = 1
    
    return result

def convAlgorithm(im,d,k):
    result = np.zeros(im.shape)
    h = np.array([[0,0,0,1,0,0,0], 
              [0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0], 
              [1,0,0,0,0,0,1], 
              [0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0], 
              [0,0,0,1,0,0,0]], np.double)
    
    red = scipy.signal.convolve2d(im[:,:,0],h, mode='same')/4
    green = scipy.signal.convolve2d(im[:,:,1],h, mode='same')/4
    blue = scipy.signal.convolve2d(im[:,:,2],h, mode='same')/4
    dist = np.sqrt((red[:,:]-im[:,:,0])**2 + (green[:,:]-im[:,:,1])**2 + (blue[:,:]-im[:,:,2])**2)  
    
    for i in range(d,im.shape[0]-d):
        for j in range(d,im.shape[1]-d):
            if dist[i,j] > k:
                result[i,j,0] = 1
                result[i,j,1] = 1
                result[i,j,2] = 1
    
    return result

image = cv2.imread("s2.jpg")
image = cv2.imread("dataset/i_0_1.jpg")
im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

im = im.astype(np.double)/255.0

plt.figure(1)
plt.imshow(im)

d = 3
k = 1
newIm = convAlgorithm(im, d, k)

plt.figure(2)
plt.imshow(newIm)

im = im.astype(np.double)/255



