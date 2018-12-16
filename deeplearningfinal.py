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
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added 
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold # needed for figuring out how well the model did
from sklearn.metrics import classification_report
# config file stuff
from configstuff import *
# import the name spaces for the nurl network models 
#from AlexNet import *
#from LeNet import *
from custom1dNet import *

def applyDownSample(arr,width):
    # run the first reducation operation, so we can know how big to make the output array
    decimated = decimate(arr[0],downsample_factor)
    newWidth = len(decimated)
    arrd = np.empty([arr.shape[0],newWidth])
    arrd[0] = decimated

    # now that we know how big our decimation is we can run the rest
    for i in np.arange(1,arrd.shape[0]):
        decimated = decimate(arr[i],downsample_factor) # down sampling factor is defined in the config file!
        arrd[i] = decimated
    
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
        rawdatarr[i] = getData(path,usefft,rawwidth)
        i += 1
    return rawdatarr

# usefull blog that this functions is based on:
# https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
# this function returns how much of the model is correct
def testModel(NNmodel,testInputs,testOutputs):
    # just to make sure the number of inputs and outputs have the same dims
    if len(testInputs) != len(testOutputs):
        print("the number of inputs doese not equal the number of outputs")
    numIterations = len(testInputs)
    #NNoutputs = np.empty(numIterations)
    NNoutputs = [] # empty list
    
#    for i in np.arange(0,numIterations):
#        #testInput_3d = np.expand_dims(testInputs[i],axis=2)
#        testInput_3d = np.reshape(-1,len(testInputs[i]),1)
    
    scores = NNmodel.predict(testInputs).round()
    
    # this part is kind of a kluge, but I need to make it work
    for i in scores:
        if i[0] == 1 and i[1] == 0:
            #print("drums")
            NNoutputs.append(0) # 0 is the code for drums
        elif i[0] == 0 and i[1] == 1:
            #print("guitar")
            NNoutputs.append(1) # one is the code for guitar
    
    correctValue = NNoutputs == testOutputs
    u,counts = np.unique(correctValue, return_counts=True)
        
    return (counts[0]/numIterations)

if __name__ == "__main__":
    # setup the folders
    datafolder = "./training"
    drumsdir = os.path.join(datafolder,"drums")
    guitardir = os.path.join(datafolder,"guitar")
    raw_width = 1058000

    # find te number of drums
    num_drums = len(os.listdir(drumsdir))
    num_guitar = len(os.listdir(guitardir))
    # generate the lables
    #lables_drums = np.repeat("drums",num_drums)
    #lables_guitar = np.repeat("guitar",num_guitar)
    # changed lables to be numbers instead of other dumb thing like string or whatever, I don't really care at this point...
    lables_drums = np.repeat(0.0,num_drums)
    lables_guitar = np.repeat(1.0,num_guitar)
    # load the raw wave data, or fft, which ever you want
    print("loading folders for training information")
    raw_drums = loadFolder(drumsdir,num_drums,use_fft,raw_width)
    raw_guitar = loadFolder(guitardir,num_guitar,use_fft,raw_width)
    # apply the down sample
    if do_downsample:
        print("running down sample")
        raw_drums = applyDownSample(raw_drums,raw_width)
        raw_guitar = applyDownSample(raw_guitar,raw_width)

    # width is the shape that we're telling the NN the input is going to be
    width = None
    if raw_drums.shape[1] == raw_guitar.shape[1]:
        width = raw_drums.shape[1]
    
    classes = 2
    # setup up the network
    NNmodel = custom1DNet(width,classes)
    #NNmodel.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    NNmodel.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    #NNmodel.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    NNmodel.summary()
    
    # do a train test split to split into validation and training data
    print("doing train test split")
    train_drums,test_drums,train_guitar,test_guitar = train_test_split(raw_drums,raw_guitar,test_size=0.25, random_state=42)
    NNinput = np.append(train_drums,train_guitar,axis=0) # inputs for training
    NNinput_test = np.append(test_drums,test_guitar,axis=0) # inputs for testing
    # inputs to the NN model have to be expaned to 3 dementions for Keras
    #NNinput_3d = NNinput.reshape(-1,len(NNinput),1) # the input for a Keras seqnetal layer has to be 3d.  So we're making it 2d
    NNinput_3d = np.expand_dims(NNinput,axis=2)
    NNinput_test_3d = np.expand_dims(NNinput_test,axis=2) 
    trainingOutputLables = np.append(lables_drums[:len(train_drums)],lables_guitar[:len(train_guitar)])
    testingOutputLables = np.append(lables_drums[:len(test_drums)],lables_guitar[:len(test_guitar)])
    #numerical_lables = convertCatigories(trainingOutputLables,catigoryies)
    # now we just do the deed
    if training == True:
        print("now training...")
        #NNmodel.fit(NNinput,trainingOutputLables.reshape(6,1),epochs=train_epochs,verbose=1)
        csv_logger = CSVLogger(csvfilename) # csvfilename defined in config file name
        NNmodel.fit(NNinput_3d,trainingOutputLables,epochs=train_epochs,verbose=1,callbacks=[csv_logger])
    else:
        print("training option is turned off")
    
    # save the trined model to a file
    if save_trained_model:
        print("saving trained modle to a file")
        NNmodel.save(savedFileName)

    #loadedNN = None
    if not save_trained_model and Load_trained_model:
        print("loading pre saved model and evuating how well it trained")
        loadedNN = load_model(savedFileName)
        if testmodel:
           scores= loadedNN.evaluate(NNinput_test_3d,testingOutputLables)
           print(scores)