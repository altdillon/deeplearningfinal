#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 03:20:19 2018

@author: dillon
"""

from pytube import YouTube # pip3 install pytube
import datetime

# do a test download of an audio stream
video_url = "https://www.youtube.com/watch?v=QNCIGl1OpMs" # documery on sexual warfare in ww2. It's wonderfull  
#url2 = "https://youtu.be/JUbMWtUyIIE"
url2 = "https://youtu.be/9ZEURntrQOg?list=RD9ZEURntrQOg"
yt_video = YouTube(url2)
#print(yt_video.title)
audio_streams = yt_video.streams.filter(only_audio=True).all() # download audio only streams
first_stream = audio_streams[0]
print("downloading video audo from: ",yt_video.title)

#global oldTs
oldTs = datetime.datetime.now().timestamp() 

def show_progress(stream, chunk, file_handle, bytes_remaining):
    global oldTs
    ts = datetime.datetime.now().timestamp() 
    if ts - oldTs > 3:
        percentleft = bytes_remaining / first_stream.filesize * 100
        percentLoaded = 100 - percentleft
        print(percentLoaded)
        oldTs = datetime.datetime.now().timestamp() 
    return

yt_video.register_on_progress_callback(show_progress)
print("starting download")
first_stream.download() # start the download

playlists = {} # define an empty dictionary 