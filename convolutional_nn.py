#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:09:08 2016

@author: Sriram
"""

import os
os.chdir('/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition') #change to home dir
import cv2
from dataExtract import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random


loc = '/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition/data' # set this to folder with data

charDict = readInData(loc, asUser=False) # 169 chars

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
charDict = readInData(loc, asUser=False, dims = [28, 28]) # 169 chars
#userDict = readInData(loc, asUser=True,  dims = [meanH, meanW]) # 143 users

checkifSameShape(charDict) # True


## count number of images. We have 45217
count = 0
for key in charDict.keys():
    for img in charDict[key]:
        count += 1
        
# Add fake images
moreDataCharDict = generateFakeData(charDict, 1)
#moreDataUserDict = generateFakeData(userDict)

X_char, y_char = genLabelsData(moreDataCharDict, oneHot=False)
#X_user, y_user = genLabelsData(moreDataUserDict)


# split into test and train data
y_char_1hot = toOneHot(y_char)[0] # we now generate the one_hot vectors for y


X_char_train, X_char_test, y_char_train, y_char_test =\
 train_test_split(X_char,y_char_1hot,test_size=0.30, random_state=99,stratify=y_char)

  
# we shuffle the training indices
indices = list(range(X_char_train.shape[0]))
random.shuffle(indices)
X_char_train = X_char_train[indices]
y_char_train = y_char_train[indices]

indices = list(range(X_char_test.shape[0]))
random.shuffle(indices)
X_char_test = X_char_test[indices]
y_char_test = y_char_test[indices]


# Parameters
miniBatchSize = 100
learning_rate = 0.001
training_iters = 150000
batch_size = 100
display_step = 100

# Network Parameters
n_input = X_char_train.shape[1]  # featurs
n_output = y_char_train.shape[1] # classes
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input]) # input holder
y = tf.placeholder(tf.float32, [None, n_output]) # output holder
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def feed_forward(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1]) # reshaped to an 30 x 30 image

    # Convolution Layer
    conv1 = conv2d(x, weights['c1'], biases['c1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['c2'], biases['c2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['d1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['d1']), biases['d1'])
    fc1 = tf.nn.relu(fc1)
    
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'c1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'c2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 166 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_output]))
}

biases = {
    'c1': tf.Variable(tf.random_normal([32])),
    'c2': tf.Variable(tf.random_normal([64])),
    'd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
pred = feed_forward(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 0), tf.argmax(y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init) # initializing all the variables

    step = 1
    
    batch_start_idx = 0
    batch_end_idx = miniBatchSize # since minibatch size is 256
    
    lossLst = []
    accLst = []
    
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x = X_char_train[batch_start_idx:batch_end_idx] 
        batch_y = y_char_train[batch_start_idx:batch_end_idx] 
    
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            
            lossLst.append(loss)
            accLst.append(acc)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        batch_start_idx = batch_end_idx 
        batch_end_idx += miniBatchSize
        if batch_end_idx >= X_char_train.shape[0]:
            batch_start_idx = 0
            batch_end_idx = miniBatchSize
        
    print("Optimization Finished!")
    
    # Calculate accuracy for the test images 
    print("Testing Accuracy:", \
       accuracy.eval(feed_dict={x: X_char_test[:20000], # we only use the first 20000
                                      y: y_char_test[:20000],
                                      keep_prob: 1.}))
    
    



