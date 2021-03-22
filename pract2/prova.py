# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:46:51 2021

@author: limue
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

img = cv2.imread('service-pnp-prok-01000-01024r.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure(1)
plt.imshow(img, 'gray')

img_resize = np.delete(img, -1, 0)

plt.figure(2)
plt.imshow(img_resize, 'gray')

size = int(np.shape(img_resize)[0]/3)

img_3_1 = img[0:size,:]
img_3_2 = img[size:size*2,:]
img_3_3 = img[size*2:size*3,:]

plt.figure(3)
plt.imshow(img_3_1, 'gray')
plt.figure(4)
plt.imshow(img_3_2, 'gray')
plt.figure(5)
plt.imshow(img_3_3, 'gray')

new_img = np.zeros((size,np.shape(img)[1],3))
new_img[:,:,0] = img_3_1
new_img[:,:,1] = img_3_2
new_img[:,:,2] = img_3_3

plt.figure(6)
plt.imshow(new_img)
