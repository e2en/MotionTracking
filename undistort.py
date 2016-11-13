import numpy as np
import cv2
import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def chdirectory (directory): #changes working directory 
    os.chdir(directory)
    f = open('dir.txt', 'r')
    lines = f.readlines()
    os.chdir(lines[0])
    return

def rddirectory (types): #returns pre-defined file directories
    images = []
    for files in types:
        images.extend(glob.glob(files))
    return images

def rdCalibrationResults (): #returns calibration matrix and distortion vector
    f = open ( 'data.txt' , 'r')
    l = []
    l = [ line.split() for line in f]
    l= map(float, l[0])+map(float, l[1])+map(float, l[2])+map(float, l[3])+map(float, l[4])+map(float, l[5])+map(float, l[6])+map(float, l[7])+map(float, l[8])+map(float, l[9]);
    mtx=np.matrix([[l[0], l[1], l[2]],[0, l[3], l[4]],[0, 0, 1]])
    dist=np.matrix([l[5], l[6], l[7], l[8], l[9]])
    return mtx, dist

def undistortion (image, height, weight): #returns undistort gray scale image 
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(weight,height),1,(weight,height))
    # undistort
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)    
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    # convert to gray-scale
    grayscale = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    return grayscale
    
directory="/Users/eren/Desktop/ComputerVision/MotionTracking"
chdirectory(directory)

types = ('*.jpg', '*.JPG')
images=rddirectory(types)

mtx, dist=rdCalibrationResults()

for fname in images:

    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    gray=undistortion(img,h,w)
    
cv2.destroyAllWindows()