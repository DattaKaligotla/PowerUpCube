import numpy as np
import cv2
from time import sleep

cap = cv2.VideoCapture(1)

while(True):
    ret, img = cap.read()
    '''if img == None:
        continue'''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.blur(hsv,(5,5))
    #lower_range = np.array([59, 45, 45], dtype=np.uint8)
    #upper_range = np.array([80, 100, 100], dtype=np.uint8)
    lower_range = np.array([20, 100, 100], dtype=np.uint8)
    upper_range = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(blur, lower_range, upper_range)
    thresh = 127
    maxValue = 255

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)

    
    cv2.waitKey(0)
    cv2.imshow('image', img)
    cv2.imshow('Objects Detected',mask)
    cv2.waitKey(0)
# When everything done, release the capture
#cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
