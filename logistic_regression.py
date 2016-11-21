#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:13:58 2016

@author: Sriram
"""

import os
os.chdir('/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition') #change to home dir
from dataExtract import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random


loc = '/Users/Sriram/Desktop/DePaul/Telugu-Char-Recognition/data' # set this to folder with data

# re-run function with new sizes
charDict = readInData(loc, asUser=True, dims = [28, 28]) # 166 chars

checkifSameShape(charDict) # True

# Add fake images
moreDataCharDict = generateFakeData(charDict,5)

# creating a labels and data matrix from these dictionaries

X_char, y_char = genLabelsData(moreDataCharDict, oneHot=False)
    
# split into test and train data
y_char_1hot = toOneHot(y_char)[0] # we now generate the one_hot vectors for y


X_char_train, X_char_test, y_char_train, y_char_test \
= train_test_split(X_char,y_char_1hot,test_size=0.30, random_state=99,stratify=y_char)
  
# we shuffle the training indices
indices = list(range(X_char_train.shape[0]))
random.shuffle(indices)
X_char_train = X_char_train[indices]
y_char_train = y_char_train[indices]

miniBatchSize = 100
epochs = 15
learning_rate = 0.01

n_input = X_char_train.shape[1]
n_output = y_char_train.shape[1] # same as number of classes
# we have no hidden layers in this case


n_test_samples = y_char_test.shape[0]

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_output]) # this is holder for the actual output

# Initialize the weights and biases
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))

pred = tf.nn.sigmoid(tf.matmul(x, W) + b) # using softmax activation

# we use cross entropy cost
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))

# change Ws and baises to reduce cross_entropy cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init) # initializing all the variables

    costLst = []
    # training starts
    for i in range(epochs): #run this for 100 epochs
        batch_start_idx = 0
        batch_end_idx = miniBatchSize # since minibatch size is 100
        
        cost = 0
        
        for _ in range(X_char_train.shape[0]//miniBatchSize):
    
            batch_x = X_char_train[batch_start_idx:batch_end_idx]
            batch_y = y_char_train[batch_start_idx:batch_end_idx]
            _1_, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            costLst.append(c)
            #print(batch_start_idx)
            batch_start_idx = batch_end_idx 
            batch_end_idx += miniBatchSize
    
            cost += c
            
            if batch_end_idx >= X_char_train.shape[0]: break
    
        avg_cost = cost / miniBatchSize
        print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    z = correct_prediction.eval({x: X_char_test, y: y_char_test})
    
    print("Accuracy:", accuracy.eval({x: X_char_test, y: y_char_test}))
    










