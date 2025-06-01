from cv2 import VideoCapture, imshow, applyColorMap, COLOR_BGR2GRAY, imwrite, cvtColor, INTER_AREA, resize, waitKey
import time
import os

savingNum = 9


dir = os.path.join("learning_dataset", str(savingNum))
os.mkdir(dir)
cam = VideoCapture(0)

for index in range(101):
    ret, frame = cam.read()
    if ret == True:
        print(dir)
        dir = os.path.join("learning_dataset", str(savingNum), str(index)+".png")
        gray = cvtColor(frame, COLOR_BGR2GRAY)
        small = resize(gray, (28,28), interpolation=INTER_AREA)
        imwrite(dir, small)
        imshow('input', frame)
        waitKey(10)
    time.sleep(0.5)
        
