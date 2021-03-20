import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def cut(img):
    size = int(np.shape(img)[0]/3)
    img_3_1 = img[0:size,:]
    img_3_2 = img[size:size*2,:]
    img_3_3 = img[size*2:size*3,:]
    return img_3_1, img_3_2, img_3_3

def getFFTconv(reference, image):
    convRef = signal.fftconvolve(reference, reference[::-1,::-1], mode='same')
    centerRef = np.unravel_index(np.argmax(convRef), convRef.shape)
    conv = signal.fftconvolve(reference, image[::-1,::-1], mode='same')
    center = np.unravel_index(np.argmax(conv), conv.shape)
    diff = [centerRef[0] - center[0], centerRef[1] - center[1]]
    new = np.roll(image,diff[0],axis=0)
    new = np.roll(new,diff[1],axis=1)

    return new

def getCrossCorr(reference, image):
    corr = signal.correlate2d(reference, image, mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    
    new = np.roll(image,x,axis=0)
    new = np.roll(new,y,axis=1)
    
    return new

def alignImages(ref, im2, im3, alignType):
    shape = ref.shape
    result = np.zeros((shape[0],shape[1],3))
    if alignType == 0:
        result[:,:,0] = ref
        result[:,:,1] = getFFTconv(ref, im2)
        result[:,:,2] = getFFTconv(ref, im3)
    elif alignType == 1:
        result[:,:,0] = ref
        result[:,:,1] = getCrossCorr(ref, im2)
        result[:,:,2] = getCrossCorr(ref, im3)
    
    return result

# ----------------------------------------------------------------------------

test1 = cv2.imread('img/test1.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure(1)
plt.imshow(test1,'gray')

test1_R, test1_G, test1_B = cut(test1)

plt.figure(2)
plt.imshow(test1_R,'gray')
plt.figure(3)
plt.imshow(test1_G,'gray')
plt.figure(4)
plt.imshow(test1_B,'gray')

result_test1 = alignImages(test1_R, test1_G, test1_B, 1)

plt.figure(5)
plt.imshow(result_test1.astype(np.uint8))





