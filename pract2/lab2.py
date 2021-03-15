import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from skimage.registration import phase_cross_correlation


def getCenteredImages(reference, image):
    convRef = signal.fftconvolve(reference, reference, mode='same')
    centerRef = np.unravel_index(np.argmax(convRef), convRef.shape)
    
    conv = signal.fftconvolve(reference, image, mode='same')
    center = np.unravel_index(np.argmax(conv), conv.shape)
    
    diff = [centerRef[0] - center[0], centerRef[1] - center[1]]
    print(reference.shape)
    print(diff)
    
    new = np.zeros((reference.shape[0], reference.shape[0]), dtype=float)
    if diff[0]>=0:
        if diff[1]>=0:
            new[diff[0]:, diff[1]:] = images[0:940-diff[0],0:940-diff[1],1]




    return new

###############################################################################

imgGray = cv2.imread('img/jhinGray.jpg', cv2.IMREAD_GRAYSCALE)

size = (640, 640)

color = [0, 0, 0]

imgCenter = cv2.copyMakeBorder(imgGray, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=color)
imgTopLeft = cv2.copyMakeBorder(imgGray, 0, 300, 0, 300, cv2.BORDER_CONSTANT, value=color)
imgTopRight = cv2.copyMakeBorder(imgGray, 0, 300, 300, 0, cv2.BORDER_CONSTANT, value=color)
imgBotLeft = cv2.copyMakeBorder(imgGray, 300, 0, 0, 300, cv2.BORDER_CONSTANT, value=color)
imgBotRight = cv2.copyMakeBorder(imgGray, 300, 0, 300, 0, cv2.BORDER_CONSTANT, value=color)

images = np.zeros((imgCenter.shape[0], imgCenter.shape[0], 5 ), dtype=float)
images[:,:,0] = imgCenter
images[:,:,1] = imgTopLeft
images[:,:,2] = imgTopRight
images[:,:,3] = imgBotLeft
images[:,:,4] = imgBotRight

newImages = np.zeros((imgCenter.shape[0], imgCenter.shape[0], 3 ), dtype=float)
newImages[:,:,1] = images[:,:,0]
newImages[:,:,1] = getCenteredImages(images[:,:,0], images[:,:,1])
newImages[:,:,2] = getCenteredImages(images[:,:,0], images[:,:,2])

plt.figure(1)
plt.imshow(images[:,:,0],'gray')
plt.figure(2)
plt.imshow(images[:,:,1],'gray')
plt.figure(3)
plt.imshow(images[:,:,2],'gray')
plt.figure(4)
plt.imshow(images[:,:,3],'gray')
plt.figure(5)
plt.imshow(images[:,:,4],'gray')