#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
#from keras.models import load_model # added
#import matplotlib.pyplot as plt
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from keras.regularizers import l2


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 08:06:03 2018
@author: dillon
"""

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
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added 
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# loads all the files from a folder and then returns them as a numpy array
# https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
# https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy/47044303#47044303
#def loadfolder(path):
#    # count the number of files in the folder 
#    filecount = 0  
#    for fn in os.listdir(path):
#        filecount += 1
#    
#    index = 0 # index value for file names
#    emptyshape = (filecount,196608) # length of a 1d array of stuff for this project
#    imgarray = np.empty(emptyshape,dtype=np.int32)
#    for image_path in glob.glob(path+"/*.png"):
#        image = imageio.imread(image_path)
#        flatimage = image.flatten() # make than image 1 dementional 
#        imgarray[index] = flatimage
#        index += 1
#        
#    return imgarray
#
#def loadfolder(path):
#    filecount = len(os.listdir(path)) # counte the number of files in a folder
#    # load the images, quick and dirty like
#    emptyshape = (filecount,256,256,3)
#    data = np.empty(emptyshape)
#    index = 0
#    for image_path in glob.glob(path+"/*.png"):
#        image = imageio.imread(image_path)
#        data[index] = image
#        index+=1
#        
#    return data # return the data 
    

# note, this is my instance of the alenet
# the         
def AlexNet(width, height, depth, classes):
    model = Sequential()
    inputshape = (width,height,depth)
    inputvolume = width*height*depth
    # first layer 
    model.add(Conv2D(filters=96, input_shape=inputshape, kernel_size=(11,11), strides=(4,4), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # second layer, max pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")) # in powerpoint pool size is 3,3
    # thierd layer, second convultion
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="valid")) # kernal used to be 11x11
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # skipping normalization, for now
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")) # again pool size is 3,3 in power point
    # fourth layer, convultional layer 
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # fith layer convultional layer 
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # max pooling...
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    # at this point we need to start getting ready for the interconnected layer 
    model.add(Flatten()) # flatten from a 3d shape into a 2d shape
    # fully connected layer, poweroint has two of these
    model.add(Dense(4096, input_shape=(inputvolume,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.4)) # to prevent overfitting
    
    model.add(Dense(4096,))
    model.add(Activation("relu"))
    model.add(Dropout(0.4)) # to prevent overfitting
    
    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    #model.add(BatchNormalization())
    
    model.add(Dense(classes)) # should have 3 classes, for some reason only works with 2
    #model.add(Activation("relu"))
    model.add(Activation("softmax"))
    return model
