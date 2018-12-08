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
from configstuff import *
# import the name spaces for the nurl network models 
from AlexNet import *
from LeNet import *

# class or struct for storing training a testing information
class Instrument:
    def __init__(self,label=None,wavdata=None,duration=0):
        self.label = label
        self.wavdata = wavdata # this had better be a numpy array !
        self.fftdata = fft(wavdata) # take the fft of the incoming data
    def getDict(self):
        instdata = {
            "label" : self.label,
            "wavarray" : self.wavdata.tobytes(), # this needs to be saved as a numpy array
            "duration" : self.duration
        }
        
        return instdata
    
    # maybe add a method to show plot data? 
    
# functions for dealing with wav files 
# find all the wav files in a folder 
def findWavFiles(folder):
    path = folder + "/*.wav" # all the wav files, I'm too anal about relitive links to not include ./ 
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

# find a folder with the name of that catigory,return None if nothing is found 
# this uses glob. FOR THE GLORY OF GLOB!
def searchCatigoryFolders(catigory,folder=""):
    folders = glob.glob(catigory)
    if len(folders) == 0: # if nothing was found matching the name in catigory
        return None
    elif len(folders) == 1 and folders[0] == catigory: # if one item was found and its name matches the name of the catigory
        return "."+folder+"/"+folders[0]     
        
# main function
if __name__ == "__main__":
    print("deep learning final project")
    print("Fall 2018, John C, Dillon H")
    print("\n")

    
    if testing == True: # testing flag defined in config file
        print("loading test audio...")
        # do a test load from the devlopment audio file
        teststuff = loadAudio("testing_devlopment","testing stuff")
        print("current Nurl network: ",currentNetwork)
        print("searching for devlopment catigory")
        devwavfolder = "testing_devlopment" # we're gonna treat this as a catigory
        devfolder = searchCatigoryFolders(devwavfolder)
        # see if we can load instruemtns from this catigory
        devinsturments = loadAudio(devfolder,"testdata_donotuse")
        print("number of catigories loaded: ",len(devinsturments))
    
    if testing == False and training == True:
        print("training mode...")
        print("loading training data")
        
        # load up the insturment objects 
        trainingData_catigories = [] # empty list to store the folder paths for the training data 
        for lables in catigoryies:
            labelfolder = searchCatigoryFolders(traingFolder+lables) # search for a folder with that label name
            if labelfolder != None: # if there's something found 
                print("loading label",labelfolder)
                trainingData_catigories.append(loadAudio(labelfolder,lables))
            
        # ok, we're now ready to train!
        
        
    if testing == False and training == False:
        print("displaying output graphs")
    
    pass # baiscly a nop
