import cv2
import numpy as np
import glob
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
kernel = np.ones((8,8),np.uint8)
loc=[]
ii=0

os.chdir("/Users/eren/Desktop/ComputerVision/MotionTracking")
f = open('dir.txt', 'r')
lines = f.readlines()
os.chdir(lines[0])

types = ('*.jpg', '*.JPG')
images = []

for files in types:
    images.extend(glob.glob(files))
    
for fname in images:
    
    img = cv2.imread(fname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255]) 
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    circles = cv2.HoughCircles(mask,cv2.cv.CV_HOUGH_GRADIENT,2,10, param1=250,param2=40,minRadius=0,maxRadius=20)
    
    circles = np.uint16(np.around(circles))
    loc.append(ii)
    loc[ii]=circles[:,:]
    
    for i in circles[0,:]:
        # draw the outer circlecv
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imwrite(str(ii) + '.png',img)
    
    ii=ii+1
    
cv2.destroyAllWindows()