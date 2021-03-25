import os
import time
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
    referenceMean = reference.astype(np.double) - reference.mean().astype(np.double) 
    imageMean = image.astype(np.double) - image.mean().astype(np.double)
    
    convRef = signal.fftconvolve(referenceMean, referenceMean[::-1,::-1], mode='same')
    centerRef = np.unravel_index(np.argmax(convRef), convRef.shape)
    conv = signal.fftconvolve(referenceMean, imageMean[::-1,::-1], mode='same')
    center = np.unravel_index(np.argmax(conv), conv.shape)
    diff = [center[0] - centerRef[0], center[1] - centerRef[1]]
    
    new = np.roll(image,diff[0],axis=0)
    new = np.roll(new,diff[1],axis=1)
    return new

def getCrossCorr(reference, image):
    referenceMean = reference.astype(np.double) - reference.mean().astype(np.double) 
    imageMean = image.astype(np.double) - image.mean().astype(np.double)
    
    corrRef = signal.correlate2d(referenceMean, referenceMean, boundary='symm', mode='same')
    yRef, xRef = np.unravel_index(np.argmax(corrRef), corrRef.shape)
    
    corr = signal.correlate2d(referenceMean, imageMean, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    
    new = np.roll(image,x-xRef,axis=1)
    new = np.roll(new,y-yRef,axis=0)

    return new

def getCorr(reference, image):
    referenceMean = reference.astype(np.double) - reference.mean().astype(np.double) 
    imageMean = image.astype(np.double) - image.mean().astype(np.double)
    
    corrRef = signal.convolve2d(referenceMean, referenceMean[::-1,::-1], boundary='symm', mode='same')
    yRef, xRef = np.unravel_index(np.argmax(corrRef), corrRef.shape)
    
    corr = signal.convolve2d(referenceMean, imageMean[::-1,::-1], boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    
    new = np.roll(image,x-xRef,axis=1)
    new = np.roll(new,y-yRef,axis=0)
    
    return new

def getFourierPhase(reference, image): 
    return 0

def processEdges(image):
    return 0

def removeNoise(image):
    return 0
    
def alignImages(ref, im2, im3, alignType, name):
    shape = ref.shape
    result = np.zeros((shape[0],shape[1],3))
    t=time.time()
    if alignType == 0:
        result[:,:,0] = ref
        result[:,:,1] = getFFTconv(ref, im2)
        result[:,:,2] = getFFTconv(ref, im3)
    elif alignType == 1:
        result[:,:,0] = ref
        result[:,:,1] = getCrossCorr(ref, im2)
        result[:,:,2] = getCrossCorr(ref, im3)
    elif alignType == 2:
        result[:,:,0] = ref
        result[:,:,1] = getCorr(ref, im2)
        result[:,:,2] = getCorr(ref, im3)
    elapsed=time.time()-t
    print(name + ': Elapsed time is '+str(elapsed)+' seconds')
    
    return result

def saveImg(img, name, extension):
    n = os.path.splitext(name)[0] + "_color" + extension
    cv2.imwrite(n, img)
    return 0

# ----------------------------------------------------------------------------

# path = "img/problemes/"
# imageSet = "p"
path = "img/set/"
imageSet = "s"
extension = ".jpg"
alignType = 0

files = os.listdir(path)

for i in range(1,len(files)):
    name = path + imageSet + str(i) + extension
    img =  cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    r, g, b = cut(img)
    result = alignImages(r,g,b,alignType, name)
    saveImg(result, path + "results/"+ imageSet + str(i), extension)
    plt.figure(i)
    plt.imshow(result.astype(np.uint8))


