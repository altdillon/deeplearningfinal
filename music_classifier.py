#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 03:12:18 2018

@author: dillon
"""

from pydub import AudioSegment
import os
import glob # for the glory of glob
# import keras for the convultional stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added 
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
# import audo stuff 
from pydub import AudioSegment
import json 
# import the name spaces for the nurl network models 
from AlexNet import *
from LeNet import *

class instrument:
    def __init__(self,label=None,wavdata=None,duration=0):
        self.label = label
        self.wavdata = wavdata # this had better be a numpy array !
        self.duration = duration
    def getDict(self):
        instdata = {
            "label" : self.label,
            "wavarray" : self.wavdata.tobytes(), # this needs to be saved as a numpy array
            "duration" : self.duration
        }
        
        return instdata
    
# loads audio and training data from a file
# returns a list of insterment objects 
def loadAudio(filepath):
    pass

def saveAudio(filepath,audiolist):
    pass
        
# main function
if __name__ == "__main__":
    pass
