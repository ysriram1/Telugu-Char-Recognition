# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import cv2

# char info: http://lipitk.sourceforge.net/datasets/teluguchar/TeluguChar.pdf

os.chdir('C:/Users/syarlag1/Desktop/Telugu-Char-Recognition') #change to home dir

loc = 'C:/Users/syarlag1/Desktop/Telugu-Char-Recognition/data' # set this to folder with data

def readInData(loc, asUser=False, dims = []):
    '''returns a dictionary reads in the images from the main data folder
    by traversing all the subfolders. 
    The keys of the dictionary are dictated by the asUser variable: 
    True, the keys are the users. False: the keys are
    the characters.
    dims
    '''
    h,w = dims
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
                if charName not in charDict.keys():
                    charDict[charName] = []
                    charDict[charName].append(tempImgMat)
                else:
                    charDict[charName].append(tempImgMat)
            os.chdir('./..') # moving back to the top folder
    return charDict
    
def checkifSameShape(imgDict):
    ''' returns True if all images are the same shape '''
    checkShape = imgDict[charDict.keys()[0]][0].shape
    
    for key in imgDict.keys():
        count = 0
        for img in imgDict[key]:
            h,w = img.shape
            if h != checkShape[0] or w != checkShape[1]:
                count += 1
    if count == 0: return True
    if count > 0: return False

def generateFakeData(imgDict):
    '''return Dict where for each image in the dataset there is fake data 
    generated through a randomized mechanism.
    '''
    for key in imgDict:
        for img in imgDict[key]:
            randomPick = np.random.choice([1,2,3,4,5])
                        


#####################

charDict = readInData(loc, asUser=False) # 169 chars
userDict = readInData(loc, asUser=True) # 143 users

# check to see if all the images are the same dimensions.
## If not interpolate to the same size
# iterating through one of the dictionaries for the check

checkifSameShape(charDict)

# Dimensions dont match for most images.

# the max h and w values
maxH = 0
maxW = 0
Hs = []
Ws = []

for key in charDict.keys():
    for img in charDict[key]:
        h,w = img.shape
        Hs.append(h); Ws.append(w)
        if h > maxH: maxH = h
        if w > maxW: maxW = w
# maxH = 737 and maxW = 769. Very high. So resizing all imgs to meanH and meanW.
meanH = int(np.mean(Hs)); meanW = int(np.mean(Ws)) #229, 235

# re-run function with new sizes
charDict = readInData(loc, asUser=False, dims = [meanH, meanW]) # 169 chars
userDict = readInData(loc, asUser=True,  dims = [meanH, meanW]) # 143 users

checkifSameShape(charDict) # True

## Recreating more data for each user
np.random.choice([1,2,3,4])
































