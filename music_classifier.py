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


# class for music catigory
class MusicCatigory:
    def __init__(self,directory):
        self.loadedSongs = 0
        self.path = directory
        self.wav_files = [] # empty list dude 
        
    def loadStuff(self):
        for audio_path in glob.glob(self.path+"/*.wav"):
            pass # nop boi
            self.wav_files.append(AudioSegment.from_wav(audio_path))
            
# main function
if __name__ == "__main__":
    print("hello world")
    a = MusicCatigory("./music/classical")
    a.loadStuff()
    pass