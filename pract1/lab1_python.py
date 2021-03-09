# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:56:31 2021

@author: limue
"""

import os
import skimage.io
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

# Tasca 1 - Carregar el dataset i dividir-lo en
# dues parts iguals, una per modelar el fons i
# l’altra per detectar [+ 0.5]

path = "highway/input/"
files = os.listdir(path)

train = []
test = []

for i in files[1050:1200]:
    fname = path + i
    train.append(skimage.io.imread(fname, as_gray=True))
    
print(np.array(train).shape)
    
for i in files[1201:1351]:
    fname = path + i
    test.append(skimage.io.imread(fname, as_gray=True))
print(np.array(test).shape)
    
# Tasca 2 - Calcular la mitjana i desviació
# estàndard [+ 0.5]

m = np.mean(train, 0)
plt.imshow(m,'gray')
std = np.std(train, 0)
print(m.dtype)
plt.imshow(std,'gray')

#  Tasca 3 - Segmentar cotxes restant el model
# de fons [+ 1.0]

segmentacio = []
thr = 0.2

for i in range(150):
    img = np.abs(test[i] - m)
    segmentacio.append((img > thr).astype(np.uint8))
    
# Tasca 4 - Segmentar cotxes amb un model
# més elaborat [+ 1.0]

segmentacio_elaborada = []
#a = 1
#b = 0.2
a = 0.13
b = 0.16
thr = a*std + b

for i in range(150):
    img = np.abs(test[i] - m)
    segmentacio_elaborada.append((img > thr).astype(np.uint8))
    
plt.imshow(segmentacio_elaborada[20],'gray')
    
# Tasca 5 - Gravar un vídeo amb els resultats
# [+ 2.0]

height = 240
width = 320

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 30
video = cv2.VideoWriter('lab1VideoPython.avi', fourcc, fps, (width, height))

for i in range(150):
    image2video  = cv2.normalize(segmentacio_elaborada[i], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image2video3c = cv2.merge([image2video, image2video, image2video])
    video.write(image2video3c)
video.release()

#os.mkdir('segmentacio')
#os.mkdir('segmentacio_elaborada')

#for i in range(150):
#    plt.imsave('segmentacio/img' + str(i) + '.jpg', segmentacio[i])
#for i in range(150):
#    plt.imsave('segmentacio_elaborada/img' + str(i) + '.jpg', segmentacio_elaborada[i])
    
#frameSize = np.shape(segmentacio_elaborada[0])
#video = cv2.VideoWriter('lab1VideoPython.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize, False)
#for i in range(150):
#    img = skimage.io.imread('segmentacio_elaborada/img' + str(i) + '.jpg', as_gray=True)
#    video.write(img)
#video.release()


# Tasca 6 - Avalua la bondat dels teus resultats [+ 1.0]
   
path = "highway/groundtruth/"
files = os.listdir(path)

ground = []
    
for i in files[1200:1350]:
    fname = path + i
    ground.append(skimage.io.imread(fname, as_gray=True))
    
errorScore = 0

for i in range(150):
    ground[i] = ground[i] > 0
    compGroup = np.zeros((240, 320, 2))
    compGroup[:,:,0] = ground[i]
    compGroup[:,:,1] = segmentacio_elaborada[i]
    score = np.std(compGroup, 2)
    errorScore = errorScore + np.sum(score);
    
errorScore = errorScore / 150
