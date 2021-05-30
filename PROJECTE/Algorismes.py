# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:17:13 2021

@author: Pipo
"""
from math import sqrt
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.signal
from scipy import ndimage
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image 
import PIL 

import skimage 
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from tqdm import tqdm

def generateFilterMatrix(size,typeFilter,width):
    m = np.zeros((size, size))
    pivot = int(size/2)
    if(typeFilter == 4):
        for i in range(0,width):
            if width<=pivot:
                m[0 + i,pivot] = 1
                m[pivot,0 + i] = 1
                m[pivot,size-1-i] = 1
                m[size-1-i,pivot] = 1
    elif(typeFilter == 8):
        for i in range(0,width):
            if width<=pivot:
                m[0+i,pivot] = 1
                m[pivot,0+i] = 1
                m[pivot,size-1-i] = 1
                m[size-1-i,pivot] = 1
                m[0+i,0+i] = 1
                m[size-1-i,0+i] = 1
                m[0+i,size-1-i] = 1
                m[size-1-i,size-1-i] = 1
    else:
        for i in range(0,width):
            if width<=pivot:
                m[0+i] = np.ones(size)
                m[size-1-i] = np.ones(size)
                m[:, 0+i] = np.ones(size)
                m[:, size-1-i] = np.ones(size)
    # print(m)
    return m

def generateFilter(matrix, im, y, x):
    return 0

def customAnomalyDetector(im, filterSize, threshold, filterType, filterWidth):
    result = np.zeros(im.shape)

    h = generateFilterMatrix(filterSize, filterType, filterWidth)
    
    total = sum(sum(h))
    
    red = scipy.signal.convolve2d(im[:,:,0],h, mode='same')/total
    green = scipy.signal.convolve2d(im[:,:,1],h, mode='same')/total
    blue = scipy.signal.convolve2d(im[:,:,2],h, mode='same')/total
    dist = np.sqrt((red[:,:]-im[:,:,0])**2 + (green[:,:]-im[:,:,1])**2 + (blue[:,:]-im[:,:,2])**2) 
        
    for i in range(filterSize,im.shape[0]-filterSize):
        for j in range(filterSize,im.shape[1]-filterSize):
            if dist[i,j] > threshold:
                result[i,j,0] = 1
                result[i,j,1] = 1
                result[i,j,2] = 1
    
    return result

def globalRX(im, threshold):
    d = np.zeros((im.shape[0], im.shape[1]))
    result = np.zeros(im.shape)
    mean = np.mean(im, axis=(0, 1))
    v = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    c = varianceCovariance(v, 10000)
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            d[i,j] = sqrt(np.dot(np.dot(np.transpose(im[i,j,:] - mean), c), im[i,j,:] - mean))
            if d[i, j] > threshold:
                result[i,j,0] = 1
                result[i,j,1] = 1
                result[i,j,2] = 1

    return result

def getValuesWindow(im, y, x, inD, outD):
    vLen = (outD *2 + 1)**2 - (inD *2 +1 )**2
    v = np.zeros((vLen, 3))
    
    newVSize = outD*(outD-inD)+((outD-inD)*((outD+1)-(outD-inD)))
    
    vX = 0
    vY = 0
    a = 0
    
    for i in range(-outD, outD+1):
        for j in range(-outD, outD+1):
            
            if abs(i) > inD or abs(j)>inD:
                val = im[y+i, x+j, :]
                v[a, :] = val
                a = a + 1

            if vX < newVSize-1:
                vX = vX + 1
            else:
                vX = 0
                vY = vY + 1
                
    return v

def localRX(im, threshold, filterSize, filterWidth):
    d = np.zeros((im.shape[0], im.shape[1]))

    result = np.zeros(im.shape)
    m = np.zeros(im.shape)
    
    v = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    c = varianceCovariance(v, 10000)
    
    filterType = 0
    h = generateFilterMatrix(filterSize, filterType, filterWidth)
    total = sum(sum(h))
    
    red = scipy.signal.convolve2d(im[:,:,0],h, mode='same')/total
    green = scipy.signal.convolve2d(im[:,:,1],h, mode='same')/total
    blue = scipy.signal.convolve2d(im[:,:,2],h, mode='same')/total
    
    m[:,:,0] = red[:,:]
    m[:,:,1] = green[:,:]
    m[:,:,2] = blue[:,:] 
                
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            d[i,j] = sqrt(np.dot(np.dot(np.transpose(im[i,j,:] - m[i,j,:]), c), im[i,j,:] - m[i,j,:]))
            if d[i, j] > threshold:
                result[i,j,0] = 1
                result[i,j,1] = 1
                result[i,j,2] = 1
            
    return result

def varianceCovariance(vector, samples):
    # vector = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
    index = np.random.randint(0, len(vector), samples)
    
    newVector = np.array([vector[i] for i in index])
    average = np.mean(newVector)
    
    matrix = np.array(sum([np.outer(np.array([newVector[i] - average]), np.array(newVector[i] - average)) for i in range(len(newVector))]) / len(newVector))
    
    return np.linalg.inv(matrix)

def compareGroundtruth(res, gt, name):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    suma = gt + res
    resta = gt - res
    
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if suma[i, j] == 2:
                TP = TP + 1
            if suma[i, j] == 0:
                TN = TN + 1
            if resta[i, j] == -1:
                FP = FP + 1
            if resta[i, j] == 1:
                FN = FN + 1
    
    ACC, TPR, FPR, PPV = valoration(TP, FN, FP, TN)
    print(name +': TP:'+str(TP)+', TN:'+str(TN)+', FP:'+str(FP)+', FN:'+str(FN)+
          ', TPR:'+str(TPR)+', FPR:'+str(FPR))
    return ACC, TPR, FPR, PPV, TP, FN, FP, TN 

def valoration(TP, FN, FP, TN):
    P = TP + FN
    N = TN + FP
    
    if P > 0:
        TPR = TP / P
    else:
        TPR = 0
    if N > 0:
        FPR = FP / N
    else: 
        FPR = 0
    if (P + N) > 0:
        ACC = (TP + TN) / (P + N)
    else:
        ACC = 0
    if (TP + FP) > 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0

    return ACC, TPR, FPR, PPV

def executeLoop(searchType, aplyMedianFilter, saveRes,
                threshold, filterSize, filterType, filterWidth,
                nRows, nCols):
    
    count = 0
    model = cv2.imread('dataset/i_0_0.jpg')
    results = np.zeros((model.shape[0], model.shape[1], nRows*nCols))
    names = np.zeros((nRows*nCols, 2))
    
    vFPR = np.zeros(nRows*nCols)
    vTPR = np.zeros(nRows*nCols)
        
    totalACC = 0
    totalTPR = 0
    totalFPR = 0
    totalPPV = 0
    totalTP = 0
    totalTN = 0
    totalFP = 0
    totalFN = 0
    
    for i in range(0,nRows):
        for j in range(0, nCols):
            img = cv2.imread('dataset/i_' + str(i) + '_'+ str(j) + '.jpg')
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = im.astype(np.double)/255.0
            gnt = cv2.imread('groundtruth/g_' + str(i) + '_'+ str(j) + '.jpg')
            gt1 = cv2.cvtColor(gnt, cv2.COLOR_BGR2RGB) 
            gt1 = gt1.astype(np.double)/255.0
            gt = gt1[:,:,0]
            
            if searchType == 0:
                res = customAnomalyDetector(im, filterSize, threshold, filterType, filterWidth)
            elif searchType == 1:
                res = localRX(im, threshold, filterSize, filterWidth)
            elif searchType == 2:
                res = globalRX(im, threshold)
                
            if aplyMedianFilter:
                r = ndimage.median_filter(res[:,:,0], 3)
            else:
                r = res[:,:,0]
            
            name = str(i) + '_'+ str(j)
            ACC, TPR, FPR, PPV, TP, FN, FP, TN= compareGroundtruth(r, gt, name)
            totalACC = totalACC + ACC
            totalTPR = totalTPR + TPR
            totalFPR = totalFPR + FPR
            totalPPV = totalPPV + PPV
            totalTP = totalTP + TP
            totalTN = totalTN + TN
            totalFP = totalFP + FP
            totalFN = totalFN + FN
            
            results[:,:,count] = r
            vFPR[count] = FPR
            vTPR[count] = TPR
            names[count, 0] = i
            names[count, 1] = j
            count = count + 1
    
    totalACC = totalACC / count
    totalTPR = totalTPR / count
    totalFPR = totalFPR / count
    totalPPV = totalPPV / count
    totalTP = totalTP / count
    totalTN = totalTN / count
    totalFP = totalFP / count
    totalFN = totalFN / count
            
    if saveRes:
        for k in range(results.shape[2]):
            i =  names[k, 0]
            j =  names[k, 1]
            im = results[:,:,k]*255
            cv2.imwrite('results/rs_'+str(int(i))+'_'+str(int(j))+".jpg", im)
    
    return totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN

def runAllTests():
    t=time.time()
    f = open("results.txt", "a")
    
    searchType = 0
    aplyMedianFilter = True
    saveRes = False
    nRows = 9
    nCols = 7
    
    maxWidth = 11
    minWidth = 3
    maxK = 1.1
        
    filterTypes = [4, 8, 1]
    
    f.write(" -- Custom Anomaly Detector -- \n")
    f.write("aplyMedianFilter = TRUE \n")
    f.write('K;fSize;fWidth;fType;ResultVal;ACC;TPR;FPR;PPV;TP;FN;FP;TN\n')
    
    for k in np.arange(0, maxK, 0.1):
        for fSize in range(minWidth, maxWidth, 1):
            for fWidth in range(1, int((fSize-1)/2)+1):
                print(k, fSize, fWidth)
                for fType in filterTypes:
                    newK = round(k, 1)
                    totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN = executeLoop(
                        searchType, aplyMedianFilter, saveRes, newK, fSize, fType, fWidth,
                        nRows, nCols)
                    resultVal = (totalTPR + (1 - totalFPR))/2
                    
                    f.write(str(newK)+';'+ 
                            str(fSize)+';'+ 
                            str(fWidth)+';'+ 
                            str(fType)+';'+
                            str(resultVal)+';'+
                            str(totalACC)+';'+
                            str(totalTPR)+';'+
                            str(totalFPR)+';'+
                            str(totalPPV)+';'+
                            str(totalTP)+';'+
                            str(totalFN)+';'+
                            str(totalFP)+';'+
                            str(totalTN)+';'+
                            '\n')
                    
                    
    searchType = 1
    maxWidth = 8
    minWidth = 3
    maxK = 2.1
    
    elapsed=time.time()-t
    print('Elapsed time is '+str(elapsed)+' seconds')
    t=time.time()
    
    f.write(" -- LOCAL RX -- \n")
    f.write("aplyMedianFilter = TRUE \n")
    f.write('K;fSize;fWidth;ResultVal;ACC;TPR;FPR;PPV;TP;FN;FP;TN\n')
    
    for k in np.arange(0, maxK, 0.1):
        for fSize in range(minWidth, maxWidth, 1):
            for fWidth in range(1, int((fSize-1)/2)+1):
                print(k, fSize, fWidth)
                newK = round(k, 1)
                totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN = executeLoop(
                    searchType, aplyMedianFilter, saveRes, newK, fSize, None, fWidth,
                    nRows, nCols)
                resultVal = (totalTPR + (1 - totalFPR))/2
                    
                f.write(str(newK)+';'+ 
                        str(fSize)+';'+ 
                        str(fWidth)+';'+ 
                        str(resultVal)+';'+
                        str(totalACC)+';'+
                        str(totalTPR)+';'+
                        str(totalFPR)+';'+
                        str(totalPPV)+';'+
                        str(totalTP)+';'+
                        str(totalFN)+';'+
                        str(totalFP)+';'+
                        str(totalTN)+';'+
                        '\n')
                
    searchType = 2    
    maxK = 4.6
    
    elapsed=time.time()-t
    print('Elapsed time is '+str(elapsed)+' seconds')
    t=time.time()
    
    f.write(" -- GLOBAL RX -- \n")
    f.write("aplyMedianFilter = TRUE \n")
    f.write('K;ResultVal;ACC;TPR;FPR;PPV;TP;FN;FP;TN\n')
    
    for k in np.arange(0, maxK, 0.1):
        print(k)
        newK = round(k, 1)
        totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN = executeLoop(
            searchType, aplyMedianFilter, saveRes, newK, None, None, None,
            nRows, nCols)
        resultVal = (totalTPR + (1 - totalFPR))/2
            
        f.write(str(newK)+';'+ 
                str(resultVal)+';'+
                str(totalACC)+';'+
                str(totalTPR)+';'+
                str(totalFPR)+';'+
                str(totalPPV)+';'+
                str(totalTP)+';'+
                str(totalFN)+';'+
                str(totalFP)+';'+
                str(totalTN)+';'+
                '\n')
    
    elapsed=time.time()-t
    print('Elapsed time is '+str(elapsed)+' seconds')
    f.close()

    
    
##############################################################################

#---------------------------------------------------------------------------------------------#
# Calculate Optimal Values

#runAllTests()


#---------------------------------------------------------------------------------------------#
# Execute with Optimal Values
aplyMedianFilter = False
saveRes = True
nRows = 10
nCols = 10

# CUSTOM

t=time.time()
# totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN = executeLoop(
#                         0, aplyMedianFilter, saveRes, 0.3, 9, 8, 1,
#                         nRows, nCols)
# resultVal = (totalTPR + (1 - totalFPR))/2

# elapsed=time.time()-t
# print('Elapsed time is '+str(elapsed/(nRows*nCols))+' seconds')
# print('RESULT: ')
# print(resultVal, totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN)
# t=time.time()

# # LOCAL
# totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN = executeLoop(
#                         1, aplyMedianFilter, saveRes, 1.8, 7, 1, 1,
#                         nRows, nCols)
# resultVal = (totalTPR + (1 - totalFPR))/2

# elapsed=time.time()-t
# print('Elapsed time is '+str(elapsed/(nRows*nCols))+' seconds')

# print('RESULT: ')
# print(resultVal, totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN)


# t=time.time()
# # Global
totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN = executeLoop(
                        2, aplyMedianFilter, saveRes, 2.9, None, None, None,
                        nRows, nCols)
resultVal = (totalTPR + (1 - totalFPR))/2

elapsed=time.time()-t
print('Elapsed time is '+str(elapsed/(nRows*nCols))+' seconds')
print('RESULT: ')
print(resultVal, totalACC, totalTPR, totalFPR, totalPPV, totalTP, totalFN, totalFP, totalTN)



