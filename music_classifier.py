#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 03:12:18 2018

@author: dillon
"""

from pydub import AudioSegment
import os
import glob # for the glory of glob
# import scipy stuff for fft
from scipy.fftpack import fft
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
import json
# import the name spaces for the nurl network models 
from AlexNet import *
from LeNet import *

# class or struct for storing training a testing information
class Instrument:
    def __init__(self,label=None,wavdata=None,duration=0):
        self.label = label
        self.wavdata = wavdata # this had better be a numpy array !
    def getDict(self):
        instdata = {
            "label" : self.label,
            "wavarray" : self.wavdata.tobytes(), # this needs to be saved as a numpy array
            "duration" : self.duration
        }
        
        return instdata
    
    
# functions for dealing with wav files 
# find all the wav files in a folder 
def findWavFiles(folder):
    path = "./"+folder + "/*.wav" # all the wav files 
    return glob.glob(path)
  
def getData(filename):
    if os.path.isfile(filename):
        wavfile = AudioSegment.from_wav(filename) 
        rawdata = wavfile.raw_data
        data = np.frombuffer(rawdata,dtype=np.uint8)
        return data
    else:
        return None

# loads audio and training data from a file
# returns a list of insterment objects 
def loadAudio(filepath,instlable):
    filenames = findWavFiles(filepath) # find a list of wav files in the folder 
    loadedCatigory = [] # empty list of loaded files 
    for file in filenames: 
        rawdata = getData(file) # load the raw wav data at the path
        newinstrument = Instrument(label=instlable,wavdata=rawdata,duration=30) # note, durationg is known to be 30
        loadedCatigory.append(newinstrument) # add the insturment to the catigory
        
    return loadedCatigory # stuf

def saveAudio(filepath,audiolist):
    pass
        
# main function
if __name__ == "__main__":
    print("deep learning final project")
    print("Fall 2018, John C, Dillon H")
    print("\n\n")
    print("loading test audio...")
    
    # do a test load from the devlopment audio file
    teststuff = loadAudio("testing_devlopment","testing stuff")
    
    pass
