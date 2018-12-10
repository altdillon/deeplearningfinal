#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:31:43 2018

@author: dillon
"""

# usefull links:
# https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

# imports 
import numpy as np
#import cv2 as cv
from scipy import misc
import imageio
import glob
import os
import pickle
# import keras for the convultional stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D,Conv1D,Reshape,MaxPooling1D,GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added 
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# ignore these for now...
sample_freq = 44.1*1000
samples = 10580000
#time_periods = samples/sample_freq
time_periods = 240 # this will never change

def custom1DNet(length,classes):
    model = Sequential()
    model.add(Conv1D(32, 4, activation='relu', input_shape=(length, 1))) # first layer, feature detector
    model.add(Conv1D(16, 4, activation='relu'))
    model.add(MaxPooling1D(4))
    # after this point I'm just adding layers for good measure 
    # the number of layers at this point is pretty arbitary
    model.add(Conv1D(32, 4, activation='relu'))
    model.add(Conv1D(32, 4, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5)) # drop out to *hopefully* prevent over fitting.
    # finaly we use a dence layer to force every thing down into two layers
    model.add(Dense(classes,activation="softmax"))
    return model