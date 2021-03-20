# -*- coding: utf-8 -*-

import numpy as np

def cut(img):
    size = int(np.shape(img)[0]/3)
    img_3_1 = img[0:size,:]
    img_3_2 = img[size:size*2,:]
    img_3_3 = img[size*2:size*3,:]
    return img_3_1, img_3_2, img_3_3
