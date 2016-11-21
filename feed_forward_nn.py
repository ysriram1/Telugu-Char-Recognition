# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
os.chdir('/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition') #change to home dir
import cv2
from dataExtract import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random


loc = '/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition/data' # set this to folder with data
# 169 chars, 144 users

charDict = readInData(loc, asUser=False, dims = [28, 28])


checkifSameShape(charDict) # True


## count number of images. We have 45217
count = 0
for key in charDict.keys():
    for img in charDict[key]:
        count += 1
        
# Add fake images
moreDataCharDict = generateFakeData(charDict, 1)


# checks
os.chdir('/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition') #change to home dir

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

X_char, y_char = genLabelsData(moreDataCharDict, oneHot=False)
    
# split into test and train data
y_char_1hot = toOneHot(y_char)[0] # generate the one_hot vectors for y
X_char_train, X_char_test, y_char_train, y_char_test = \
train_test_split(X_char,y_char_1hot,test_size=0.30, random_state=99,stratify=y_char)
  
# we shuffle the training indices
indices = list(range(X_char_train.shape[0]))
random.shuffle(indices)
X_char_train = X_char_train[indices]
y_char_train = y_char_train[indices]

# shuffle test indices
indices = list(range(X_char_test.shape[0]))
random.shuffle(indices)
X_char_test = X_char_test[indices]
y_char_test = y_char_test[indices]


###############################
# Now that we have the necessary fake data added, we can proceed with learning
# learn with a simple feed forward neural network
# with 3 hidden layers

## Start TensorFlow ##
# initialize the params
miniBatchSize = 100
epochs = 25
learning_rate = 0.001

n_input = X_char_train.shape[1]
n_h1 = 100 # hidden layer 1 neurons
n_h2 = 100 # hidden layer 2 neurons
n_output = y_char_train.shape[1] # same as number of classes

n_test_samples = y_char_test.shape[0]

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_output])


def forwardProp(inputs, biases, weights):
    ''' returns the outputs. Uses ReLu as activation at hidden layers.
    Two Hidden Layers.
    '''
    l1 = tf.nn.relu(tf.add(tf.matmul(inputs, weights['l1']),biases['l1']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, weights['l2']),biases['l2']))
    outputs = tf.matmul(l2,weights['out']) + biases['out']
    return outputs
    
# randomly initialize weights and baises using a gaussian dist
weights = { 'l1': tf.Variable(tf.random_normal([n_input,n_h1])), # first layer W matrix
           'l2': tf.Variable(tf.random_normal([n_h1,n_h2])), # second layer
            'out': tf.Variable(tf.random_normal([n_h2,n_output])) # output layer
           }

biases = { 'l1': tf.Variable(tf.random_normal([n_h1])), # first layer W matrix
       'l2': tf.Variable(tf.random_normal([n_h2])), # second layer
        'out': tf.Variable(tf.random_normal([n_output])) # output layer
       }

pred = forwardProp(x,biases, weights) # predicted y arrays

# calculate the cross entropy cost
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

# change Ws and baises to reduce cross_entropy cost
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init) # initializing all the variables


    #correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    
    costLst = []; accLst = []
    # training starts
    for i in range(epochs): #run this for 100 epochs
        batch_start_idx = 0
        batch_end_idx = miniBatchSize # since minibatch size is 100
        
        cost = 0
        acc_total = 0
        
        for _ in range(X_char_train.shape[0]//miniBatchSize):
            
            noOfBatchs = X_char_train.shape[0]//miniBatchSize
    
            batch_x = X_char_train[batch_start_idx:batch_end_idx] 
            batch_y = y_char_train[batch_start_idx:batch_end_idx] 
            _i_, c, acc = sess.run([optimizer, cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y})
            #print(batch_start_idx)
            batch_start_idx = batch_end_idx 
            batch_end_idx += miniBatchSize
            cost += c; acc_total += acc
            if batch_end_idx >= X_char_train.shape[0]: break
        
        avg_cost = cost / noOfBatchs; costLst.append(avg_cost)
        avg_acc = acc_total/ noOfBatchs; accLst.append(avg_acc)
        
        print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))
    
    
    # testing
      
    print(accuracy.eval(feed_dict={x: X_char_test, y: y_char_test}))
    
sess.close()








