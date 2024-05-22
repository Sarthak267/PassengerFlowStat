import cv2 as cv
import numpy as n

cap=cv.VideoCapture(0)

while True:
    success,img=cap.read()
    cv.imshow("Video",img)
cv2.waitKey(1)