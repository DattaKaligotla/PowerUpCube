import numpy as np
from cv2 import *
from time import sleep
import scipy
from PIL import Image
cap = cv2.VideoCapture(1)
while(True):
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.blur(hsv,(5,5))
    #lower_range = np.array([59, 45, 45], dtype=np.uint8)
    #upper_range = np.array([80, 100, 100], dtype=np.uint8)
    lower_range = np.array([20, 100, 100], dtype=np.uint8)
    upper_range = np.array([30, 255, 255], dtype=np.uint8)
    rgb = cv2.inRange(blur, lower_range, upper_range)
    small = pyrDown(rgb)
    # apply grayscale
    #small = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # morphological gradient
    morph_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = morphologyEx(small, cv2.MORPH_GRADIENT, morph_kernel)
    # binarize
    _, bw = threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morph_kernel = getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # connect horizontally oriented regions
    connected = morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
    mask = np.zeros(bw.shape, np.uint8)
    # find contours
    im2, contours, hierarchy = findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours
    for idx in range(0, len(hierarchy[0])):
        rect = x, y, rect_width, rect_height = boundingRect(contours[idx])
        # fill the contour
        mask = drawContours(mask, contours, idx, (255, 255, 2555), cv2.FILLED)
        # ratio of non-zero pixels in the filled region
        r = float(countNonZero(mask)) / (rect_width * rect_height)
        if r > 0.45 and rect_height > 8 and rect_width > 8:
            rgb = rectangle(rgb, (x, y+rect_height), (x+rect_width, y), (0,255,0),1)
    Image.fromarray(rgb).show()
    '''thresh = 127
    maxValue = 255
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)'''
    cv2.waitKey(0)
    cv2.imshow('image', img)
    cv2.imshow('Objects Detected',rgb)
    cv2.waitKey(0)
# When everything done, release the capture
#cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
