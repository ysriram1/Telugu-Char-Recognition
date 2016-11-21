#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:05:29 2016

@author: Sriram
"""

import os
import numpy as np
import cv2
from copy import deepcopy

# char info: http://lipitk.sourceforge.net/datasets/teluguchar/TeluguChar.pdf

def readInData(loc, asUser=False, dims = []):
    '''returns a dictionary reads in the images from the main data folder
    by traversing all the subfolders. 
    The keys of the dictionary are dictated by the asUser variable: 
    True, the keys are the users. False: the keys are
    the characters.
    dims
    '''
    if len(dims) == 2: h,w = dims
    subFolderLst = os.listdir(loc)
    charDict = {}
    for folder in subFolderLst:

        imFilesLst = os.listdir(loc + '/' + folder)
        
        for img in imFilesLst:
            if img[-4:] != 'tiff': continue #skip if not image file
            tempImgLoc = loc + '/' + folder + '/' + img            
            # convert to grayscale
            tempImgMat = cv2.cvtColor(cv2.imread(tempImgLoc), cv2.COLOR_RGB2GRAY) 
            # resize image
            if len(dims) == 2: tempImgMat = cv2.resize(tempImgMat, (h,w))
            if asUser: # if images to be classified by user
                if folder not in charDict.keys():
                    charDict[folder] = []
                    charDict[folder].append(tempImgMat)
                else:
                    charDict[folder].append(tempImgMat)
            if not asUser:
                charName = img.split('t')[0]
                if charName in ['94','99','56']: os.chdir('./..'); continue #we dont want to consider these values
                if charName not in charDict.keys():
                    charDict[charName] = []
                    charDict[charName].append(tempImgMat)
                else:
                    charDict[charName].append(tempImgMat)
            os.chdir('./..') # moving back to the top folder
    return charDict
    
def checkifSameShape(imgDict):
    ''' returns True if all images are the same shape '''
    checkShape = imgDict[imgDict.keys()[0]][0].shape
    
    for key in imgDict.keys():
        count = 0
        for img in imgDict[key]:
            h,w = img.shape
            if h != checkShape[0] or w != checkShape[1]:
                count += 1
    if count == 0: return True
    if count > 0: return False


def addSP(im):
    ''' returns image with salt and pepper noise '''
    noise = im.copy()
    noise = cv2.randn(noise,(0),(99))
    return im + noise    
    
    
def generateFakeData(imDict, n):
    '''return Dict where for each image in the dataset there is fake data 
    generated through a randomized mechanism.
    '''
    imgDict = deepcopy(imDict)    
    kernel = np.ones((5,5), np.uint8) # used for dilating and eroding
    for key in imDict.keys():
        for img in imDict[key]:
            for _ in range(n):
                randomPick = np.random.choice([1,2,3,4,5]) #erode and median blurring removed
                if randomPick == 1: # median blurring
                    tempImg = deepcopy(img)                  
                    imgDict[key].append(cv2.medianBlur(tempImg,5))
                    
                if randomPick == 2: # average blurring
                    tempImg = deepcopy(img)
                    imgDict[key].append(cv2.blur(tempImg,(5,5)))
                    
                if randomPick == 3: # add salt and pepper noise
                    tempImg = deepcopy(img)              
                    imgDict[key].append(addSP(tempImg))
                
                if randomPick == 4: # erode
                    tempImg = deepcopy(img)
                    imgDict[key].append(cv2.erode(tempImg, kernel, iterations=1))
                
                if randomPick == 5: # dilate
                    tempImg = deepcopy(img)
                    imgDict[key].append(cv2.dilate(tempImg, kernel, iterations=1))
    return imgDict
    
    

def toOneHot(y):
    '''returns a one_hot representation of the labels. Can handle string labels as well'''
    uniqueValues = set(y)
    referenceDict = {val:idx for idx,val in enumerate(uniqueValues)}
    
    y_onehot = np.zeros([len(y),len(uniqueValues)])
    for i,val in enumerate(y):
        y_onehot[i][referenceDict[val]] = 1

    y_onehot.astype(bool)
                     
    return y_onehot, referenceDict
    
    

def genLabelsData(imDict, oneHot = True):
    '''returns the labels and the datapoints for the input dictionaries. Each 
    datapoints is an image that is flatten so that it is represented as a
    vector. Labels are outputted as one hot vectors.
    '''
    y = []
    for key in imDict.keys():
        y += [key]*len(imDict[key])
    
    y = np.array(y)
    
    if oneHot: y, ref = toOneHot(y)

    X = []
    for values in imDict.values():
        for img in values:
            img_vec = img.reshape(-1)
            X.append(img_vec)
    
    X = np.array(X)
    
    if oneHot: return X, y, ref
    
    return X, y
                          
    