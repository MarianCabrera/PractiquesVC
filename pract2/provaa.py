# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:02:31 2021

@author: limue
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# read image and part into 3 subimages

img = cv2.imread('service-pnp-prok-01000-01024r.jpg', cv2.IMREAD_GRAYSCALE)
size = int(np.shape(img)[0]/3)
img_red = img[0:size,:]
img_green = img[size:size*2,:]
img_blue = img[size*2:-1,:]

plt.figure(1)
plt.imshow(img, 'gray')
plt.figure(2)
plt.imshow(img_red, 'gray')
plt.figure(3)
plt.imshow(img_green, 'gray')
plt.figure(4)
plt.imshow(img_blue, 'gray')

# create new 3d image

new_img = np.zeros((img_red.shape[0], img_red.shape[1], 3))
new_img[:,:,0] = img_red

# calculate convolutions

rr_convolve = signal.fftconvolve(img_red, img_red)
rg_convolve = signal.fftconvolve(img_red, img_green)
rb_convolve = signal.fftconvolve(img_red, img_blue)

# get the centers of each convolution

center_rr = np.unravel_index(np.argmax(rr_convolve), rr_convolve.shape)
center_rg = np.unravel_index(np.argmax(rg_convolve), rg_convolve.shape)
center_rb = np.unravel_index(np.argmax(rb_convolve), rb_convolve.shape)

# get the displacement of red-green and red-blue
desp_rg = tuple([abs(center_rr[0] - center_rg[0]), abs(center_rr[1] - center_rg[1])])
desp_rb = tuple([abs(center_rr[0] - center_rb[0]), abs(center_rr[1] - center_rb[1])])

# displace the images

new_img[:,:,1] = np.roll(img_green, desp_rg[0], 0)
new_img[:,:,1] = np.roll(new_img[:,:,1], desp_rg[1], 1)
new_img[:,:,2] = np.roll(img_blue, desp_rb[0], 0)
new_img[:,:,2] = np.roll(new_img[:,:,2], desp_rb[1], 1)

# write the image

plt.figure(5)
plt.imshow(new_img)
cv2.imwrite("hola.jpg", new_img)

# put original images together to see if there is a difference

new_img_no_disp = np.zeros((img_red.shape[0], img_red.shape[1], 3))
new_img_no_disp[:,:,0] = img_red
new_img_no_disp[:,:,1] = img_green
new_img_no_disp[:,:,2] = img_blue
plt.figure(6)
plt.imshow(new_img_no_disp)
cv2.imwrite("hola_no_disp.jpg", new_img_no_disp)
