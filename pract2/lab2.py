import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import skimage.io


def getFFTconv(reference, image):
    convRef = signal.fftconvolve(reference, reference[::-1,::-1], mode='same')
    centerRef = np.unravel_index(np.argmax(convRef), convRef.shape)
    conv = signal.fftconvolve(reference, image[::-1,::-1], mode='same')
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
    
    print((x-xRef), (y-yRef))
    new = np.roll(image,x-xRef,axis=1)
    new = np.roll(new,y-yRef,axis=0)
    
    fig, (ax_orig, ax_template, ax_corr, ax_result) = plt.subplots(4, 1, figsize=(6, 15))
    ax_orig.imshow(reference, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_template.imshow(image, cmap='gray')
    ax_template.set_title('Template') 
    ax_template.set_axis_off() 
    ax_corr.imshow(corr, cmap='gray') 
    ax_corr.set_title('Cross-correlation') 
    ax_corr.set_axis_off() 
    ax_orig.plot(x, y, 'ro')
    ax_orig.plot(xRef, yRef, 'ro') 
    ax_result.imshow(new, cmap='gray')
    ax_result.set_title('result') 
    ax_result.set_axis_off() 
    
    fig.show()
    
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
    elif alignType == 2:
        result[:,:,0] = ref
        result[:,:,1] = getCorr(ref, im2)
        result[:,:,2] = getCorr(ref, im3)
        
    return result

def getConv(reference, image):
    
    newScale = (int(reference.shape[1]/4),int(reference.shape[0]/4))
    referenceResized = cv2.resize(reference, dsize=newScale, interpolation=cv2.INTER_CUBIC)
    imageResized = cv2.resize(image, dsize=newScale, interpolation=cv2.INTER_CUBIC)
    
    convReference = signal.convolve2d(referenceResized, referenceResized[::-1,::-1],mode='same')
    centerReference = np.unravel_index(np.argmax(convReference), convReference.shape)
    
    conv = signal.convolve2d(referenceResized, imageResized[::-1,::-1],mode='same')
    center = np.unravel_index(np.argmax(conv), conv.shape)
    
    
    diffX = (centerReference[0] - center[0] ) * 4
    new = np.roll(image,diffX,axis=0)
    diffY = (centerReference[1] - center[1] ) * 4
    new = np.roll(new,diffY,axis=1)
    
    
    return new
    

###############################################################################

imgGray = cv2.imread('img/jhinGray.jpg', cv2.IMREAD_GRAYSCALE)

size = (640, 640)

color = [0, 0, 0]

imgCenter = cv2.copyMakeBorder(imgGray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=color)
imgTopLeft = cv2.copyMakeBorder(imgGray, 0, 30, 0, 30, cv2.BORDER_CONSTANT, value=color)
imgTopRight = cv2.copyMakeBorder(imgGray, 0, 30, 30, 0, cv2.BORDER_CONSTANT, value=color)
imgBotLeft = cv2.copyMakeBorder(imgGray, 30, 0, 0, 30, cv2.BORDER_CONSTANT, value=color)
imgBotRight = cv2.copyMakeBorder(imgGray, 30, 0, 30, 0, cv2.BORDER_CONSTANT, value=color)

# images = np.zeros((imgCenter.shape[0], imgCenter.shape[0], 5 ))
# images[:,:,0] = imgCenter
# images[:,:,1] = imgTopLeft
# images[:,:,2] = imgTopRight
# images[:,:,3] = imgBotLeft
# images[:,:,4] = imgBotRight

# newImages = np.zeros((imgCenter.shape[0], imgCenter.shape[0], 5 ), dtype=float)
# newImages[:,:,0] = images[:,:,0]
# newImages[:,:,1] = getFFTconv(images[:,:,0], images[:,:,1])
# newImages[:,:,2] = getFFTconv(images[:,:,0], images[:,:,2])
# newImages[:,:,3] = getFFTconv(images[:,:,0], images[:,:,3])
# newImages[:,:,4] = getFFTconv(images[:,:,0], images[:,:,4])


## TODO cut edges

# plt.figure(1)
# plt.imshow(images[:,:,0],'gray')
# plt.figure(2)
# plt.imshow(images[:,:,1],'gray')
# plt.figure(3)
# plt.imshow(images[:,:,2],'gray')
# plt.figure(4)
# plt.imshow(images[:,:,3],'gray')
# plt.figure(5)
# plt.imshow(images[:,:,4],'gray')

imgColor = skimage.io.imread('img/sonw.jpg')
r = imgColor[:,:,0]
g = imgColor[:,:,1]
b = imgColor[:,:,2]

r1 = cv2.copyMakeBorder(r, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)
g1 = cv2.copyMakeBorder(g, 0, 10, 0, 10, cv2.BORDER_CONSTANT, value=color)
b1 = cv2.copyMakeBorder(b, 0, 10, 10, 0, cv2.BORDER_CONSTANT, value=color)
g2 = cv2.copyMakeBorder(g, 10, 0, 0, 10, cv2.BORDER_CONSTANT, value=color)
b2 = cv2.copyMakeBorder(b, 10, 0, 10, 0, cv2.BORDER_CONSTANT, value=color)

# print(r.dtype)

plt.figure(1)
plt.imshow(r1,'gray')
plt.figure(2)
plt.imshow(g1,'gray')
plt.figure(3)
plt.imshow(b1,'gray')
plt.figure(4)
plt.imshow(g2,'gray')
plt.figure(5)
plt.imshow(b2,'gray')

result = alignImages(r1, g1, b1, 2)

plt.figure(8)
plt.imshow(result.astype(np.uint8))

# result = alignImages(r1, g1, b1, 1)

# plt.figure(11)
# plt.imshow(result.astype(np.uint8))


# newScale = (int(r1.shape[1]/4),int(r1.shape[0]/4))
# refRes = cv2.resize(r1, dsize=newScale, interpolation=cv2.INTER_CUBIC)
# imgRes = cv2.resize(g1, dsize=newScale, interpolation=cv2.INTER_CUBIC)

# plt.figure(7)
# plt.imshow(r1,'gray')

# convRef = signal.convolve2d(refRes, refRes[::-1,::-1],mode='same')
# centerRef = np.unravel_index(np.argmax(convRef), convRef.shape)
# conv = signal.convolve2d(refRes, imgRes[::-1,::-1],mode='same')
# center = np.unravel_index(np.argmax(conv), conv.shape)

# print(centerRef)
# print(center)

# diffX = -(centerRef[0] - center[0] ) * 4
# new = np.roll(g1,diffX,axis=0)
# diffY = -(centerRef[1] - center[1] ) * 4
# new = np.roll(new,diffX,axis=1)

















