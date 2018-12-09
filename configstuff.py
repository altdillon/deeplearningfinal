#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:27:12 2018

@author: dillon
"""

import os

# just variables to turn stuff on and off
testing = False
training = True
classify = False
display_results = False
use_fft = False
save_trained_model = True
# current nurl networks, AlexNet,LeNet, maybe some custom stuff
currentNetwork = "CustomNet"
# list for instrument catigoryies
# As ashamed as I am to admit this, I feel that you're more qualified to edit this than I am ~Dillon
catigoryies = ["drums","guitar"]

# names of folders for training and testing
# NOTE: We don't need ./ here
traingFolder = "training"
testingFolder = "testing" 
# settings for training
train_epochs = 25 # I this is what I used for the midterm
batch_size = 32 # defult value in kears anyway