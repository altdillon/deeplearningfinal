import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


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

def loadfolder(path):
    filecount = len(os.listdir(path)) # counte the number of files in a folder
    # load the images, quick and dirty like
    emptyshape = (filecount,256,256,3)
    data = np.empty(emptyshape)
    index = 0
    for image_path in glob.glob(path+"/*.png"):
        image = imageio.imread(image_path)
        data[index] = image
        index+=1
        
    return data # return the data 
    

# lenet, from Chao's minst hand writing exsample 
def LeNet(width=1, height=1, depth=1, classes=1):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		#if K.image_data_format() == "channels_first":
		#	inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
