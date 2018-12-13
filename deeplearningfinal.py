#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:17:16 2018

@author: dillon
"""

import numpy as np
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import os
import glob # for the glory of glob
# import scipy stuff for fft
from scipy.fftpack import fft
from scipy.signal import decimate # for down sampling
# import keras for the convultional stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added 
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
# config file stuff
from configstuff import *
# import the name spaces for the nurl network models 
#from AlexNet import *
#from LeNet import *
from custom1dNet import *

def applyDownSample(arr,width):
    arrd = np.empty(width)
    for i in arr:
        arrd = yd = decimate(i,downsample_factor) # down sampling factor is defined in the config file
        
    return arrd

def convertCatigories(lables,labarr):
    numlables = np.empty(len(labarr))
    index = 0 
    
    for elem in labarr:
        for lable in lables:
            lvalue = 0
            if elem == lable:
                numlables[index] = lvalue
            else:
                lvalue += 1
        index += 1
            
    
    return numlables
    

def getData(filename,use_fft=False,rwidth=1):
    if os.path.isfile(filename):
        wavfile = AudioSegment.from_wav(filename)
        rawdata = wavfile.raw_data
        data = np.frombuffer(rawdata,dtype=np.uint8)
        if len(data) > rwidth:
            data = data[:rwidth]
        if use_fft == True:
            data = fft(data)
        return data
    else:
        return None
    
def loadFolder(foldername,numitems,usefft,rawwidth):
    i = 0
    #width = 10580000
    #width = raw_width
    #rawdatarr = np.empty([numitems,10580000])
    print("numitems: ",numitems," type: ",type(numitems)," rawwidth: ",rawwidth," rawwidth type: ",type(rawwidth))
    rawdatarr = np.empty([numitems,rawwidth])
    for file in os.listdir(foldername):
        path = os.path.join(foldername,file)
        rawdatarr[i] = getData(path,usefft,width)
        i += 1
    return rawdatarr

if __name__ == "__main__":
    # setup the folders
    datafolder = "./training"
    drumsdir = os.path.join(datafolder,"drums")
    guitardir = os.path.join(datafolder,"guitar")
    #width = 10580000
    width = 1058000
    classes = 2
    # setup up the network
    NNmodel = custom1DNet(width,classes)
    #NNmodel.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    NNmodel.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    NNmodel.summary()
    # find te number of drums
    num_drums = len(os.listdir(drumsdir))
    num_guitar = len(os.listdir(guitardir))
    # generate the lables
    #lables_drums = np.repeat("drums",num_drums)
    #lables_guitar = np.repeat("guitar",num_guitar)
    # changed lables to be numbers instead of other dumb thing like string or whatever, I don't really care at this point...
    lables_drums = np.repeat(1.0,num_drums)
    lables_guitar = np.repeat(2.0,num_guitar)
    # load the raw wave data, or fft, which ever you want
    print("loading folders for training information")
    raw_drums = loadFolder(drumsdir,num_drums,use_fft,width)
    raw_guitar = loadFolder(guitardir,num_guitar,use_fft,width)
    # apply the down sample
    if do_downsample:
        print("running down sample")
        raw_drums = applyDownSample(raw_drums,width)
        raw_guitar = applyDownSample(raw_guitar,width)
    
    # do a train test split to split into validation and training data
    print("doing train test split")
    train_drums,test_drums,train_guitar,test_guitar = train_test_split(raw_drums,raw_guitar,test_size=0.25, random_state=42)
    NNinput = np.append(train_drums,train_guitar,axis=0)
    NNinput_3d = NNinput.reshape(-1,len(NNinput),1) # the input for a Keras seqnetal layer has to be 3d.  So we're making it 2d
    #NNinput = np.expand_dims(NNinput_2d,axis=2)
    trainingOutputLables = np.append(lables_drums[:len(train_drums)],lables_guitar[:len(train_guitar)])
    #numerical_lables = convertCatigories(trainingOutputLables,catigoryies)
    # now we just do the deed
    if training == True:
        print("now training...")
        #NNmodel.fit(NNinput,trainingOutputLables.reshape(6,1),epochs=train_epochs,verbose=1)
        NNmodel.fit(NNinput_3d,trainingOutputLables,epochs=train_epochs,verbose=1)
    else:
        print("training option is turned off")
    
    
