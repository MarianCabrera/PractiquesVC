import cv2
import numpy as np
from matplotlib import pyplot as plt

pathProb = "img/cnm/"
extension = ".jpg"
numImgs = 3

name = pathProb + "image001.jpg"
img = plt.imread(name)
imgShape = img.shape

set3 = np.zeros((imgShape[0], imgShape[1], 3, numImgs))
for i in range(numImgs):
    name = pathProb + "image0"
    if i <9:
        name = name + "0"
        
    name = name + str(i+1) + extension
    img = plt.imread(name)
    set3[:,:,:,i]= img.astype(np.double)/256

plt.figure(1)
plt.imshow(set3[:,:,:,0])

x= plt.ginput(2) 

print(x)
plt.show()


    