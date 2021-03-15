import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from skimage.registration import phase_cross_correlation


imgGray = cv2.imread('img/jhinGray.jpg', cv2.IMREAD_GRAYSCALE)

size = (640, 640)

color = [0, 0, 0]
imgGrayR = cv2.copyMakeBorder(imgGray, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=color)
imgGrayG = cv2.copyMakeBorder(imgGray, 0, 300, 0, 300, cv2.BORDER_CONSTANT, value=color)
imgGrayB = cv2.copyMakeBorder(imgGray, 200, 100, 300, 0, cv2.BORDER_CONSTANT, value=color)

images = np.zeros((imgGrayR.shape[0], imgGrayR.shape[0], 3 ), dtype=float)
images[:,:,0] = imgGrayR
images[:,:,1] = imgGrayG
images[:,:,2] = imgGrayB

images[:,:,0] -= np.mean(images[:,:,0])
images[:,:,1] -= np.mean(images[:,:,1])
images[:,:,2] -= np.mean(images[:,:,2])


conv1 = signal.fftconvolve(images[:,:,0], images[:,:,0], mode='same')
conv2 = signal.fftconvolve(images[:,:,0], images[:,:,1], mode='same')
conv3 = signal.fftconvolve(images[:,:,0], images[:,:,2], mode='same')

center1 = np.unravel_index(np.argmax(conv1), conv1.shape)
center2 = np.unravel_index(np.argmax(conv2), conv2.shape)
center3 = np.unravel_index(np.argmax(conv3), conv3.shape)
print(center1)
print(center2)
print(center3)

diff1 = [center1[0] - center2[0], center1[1] - center2[1]]
print(diff1)


#plt.imshow(conv1,'gray')