import agent as a
import time
import cv2 as cv
import os
# a.detectPng("model-1", "learning_dataset", 3)
snap = cv.VideoCapture(0)
while True:
    
    a.detectCam("model-1", snap)
    time.sleep(0)    # seconds
    

snap.release()
