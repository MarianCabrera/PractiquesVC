import os
import skimage.io
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

# Tasca 1
path = "highway/input/"
files = os.listdir(path)
train = []
test = []

for i in files[1050:1200]:
    fname = path + i
    train.append(skimage.io.imread(fname, as_gray=True))  
for i in files[1201:1351]:
    fname = path + i
    test.append(skimage.io.imread(fname, as_gray=True))
    
# Tasca 2
m = np.mean(train, 0)
std = np.std(train, 0)

#  Tasca
segmentacio = []
thr = 0.2
for i in range(150):
    img = np.abs(test[i] - m)
    segmentacio.append((img > thr).astype(np.uint8))
    
# Tasca 4
segmentacio_elaborada = []
a = 0.13
b = 0.16
thr = a*std + b
for i in range(150):
    img = np.abs(test[i] - m)
    segmentacio_elaborada.append((img > thr).astype(np.uint8))
    
# Tasca 5
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

# Tasca 6
path = "highway/groundtruth/"
files = os.listdir(path)
ground = []
for i in files[1201:1351]:
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
print(errorScore)