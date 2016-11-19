# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


def addSP(im):
    ''' returns image with salt and pepper noise '''
    noise = im.copy()
    noise = cv2.randn(noise,(0),(99))
    return im + noise    
    
    
def generateFakeData(imDict):
    '''return Dict where for each image in the dataset there is fake data 
    generated through a randomized mechanism.
    '''
    imgDict = deepcopy(imDict)    
    kernel = np.ones((5,5), np.uint8) # used for dilating and eroding
    for key in imDict.keys():
        for img in imDict[key]:
            randomPick = np.random.choice([1,2,3,4,5])
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
                
def genLabelsData(imDict):
    '''returns the labels and the datapoints for the input dictionaries. Each 
    datapoints is an image that is flatten so that it is represented as a
    vector.
    '''
    y = []
    for key in imDict.keys():
        y += [key]*len(imDict[key])
    
    y = np.array(y)

    X = []
    for values in imDict.values():
        for img in values:
            img_vec = img.reshape(-1)
            X.append(img_vec)
    
    X = np.array(X)
    
    return X, y
                          


#####################
import pickle


os.chdir('C:/Users/syarlag1/Desktop/Telugu-Char-Recognition') #change to home dir

loc = 'C:/Users/syarlag1/Desktop/Telugu-Char-Recognition/data' # set this to folder with data

charDict = readInData(loc, asUser=False) # 169 chars
userDict = readInData(loc, asUser=True) # 143 users
#charDictFile = open('charData.pickle',"wb")
#userDictFile = open('userData.pickle',"wb")
#pickle.dump(charDict, charDictFile); charDictFile.close()
#pickle.dump(userDict, userDictFile); userDictFile.close()

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

## count number of images. We have 45217
count = 0
for key in charDict.keys():
    for img in charDict[key]:
        count += 1
        
# Add fake images
moreDataCharDict = generateFakeData(charDict)
moreDataUserDict = generateFakeData(userDict)



# checks
tempImg = moreDataCharDict['000'][500]
tempImg2 = moreDataCharDict['000'][520]
tempImg3 = moreDataCharDict['000'][400]
tempImg4 = moreDataCharDict['000'][235]
tempImg5 = moreDataCharDict['000'][335]
tempImg6 = moreDataCharDict['000'][4]
tempImg7 = moreDataCharDict['000'][4+267]
cv2.imwrite('t1.jpg',tempImg)
cv2.imwrite('t2.jpg',tempImg2)
cv2.imwrite('t3.jpg',tempImg3)
cv2.imwrite('t4.jpg',tempImg4)
cv2.imwrite('t5.jpg',tempImg5)
cv2.imwrite('t6.jpg',tempImg6)
cv2.imwrite('t7.jpg',tempImg7)

# creating a labels and data matrix from these dictionaries

X_char, y_char = genLabelsData(moreDataCharDict)
X_user, y_user = genLabelsData(moreDataUserDict)

# saving to a pickle file ... has not worked so far-- taking too long
np.savetxt('charX.csv',X_char,delimiter=',')
np.savetxt('charY.csv',y_char,delimiter=',')

charX = open('charX.pickle','w')
pickle.dump(X_char, charX)
charX.close()

charY = open('charY.pickle','w')
pickle.dump(y_char, charY)
charY.close()
    
    


# Now that we have the necessary fake data added, we can proceed with learning
# First we learn with a simple feed forward neural network





# Next we learn using a 4 layered CNN




# Results

























