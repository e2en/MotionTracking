import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

os.chdir("/Users/eren/Desktop/ComputerVision/MotionTracking")
f = open('dir.txt', 'r')
lines = f.readlines()
os.chdir(lines[0])

###read calibration parameters
f = open ( 'data.txt' , 'r')
l = []
l = [ line.split() for line in f]

l= map(float, l[0])+map(float, l[1])+map(float, l[2])+map(float, l[3])+map(float, l[4])+map(float, l[5])+map(float, l[6])+map(float, l[7])+map(float, l[8])+map(float, l[9]);
mtx=np.matrix([[l[0], l[1], l[2]],[0, l[3], l[4]],[0, 0, 1]])
dist=np.matrix([l[5], l[6], l[7], l[8], l[9]])
########################################################

types = ('*.jpg', '*.JPG')
images = []
for files in types:
    images.extend(glob.glob(files))

for fname in images:

    img = cv2.imread(fname)

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    ## undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    imgROI=img[220:653, 218:680]
    gray = cv2.cvtColor(imgROI,cv2.COLOR_BGR2GRAY)
    
    #ret, corners = cv2.findChessboardCorners(gray, (2,2),None)
    corners=cv2.cornerHarris(gray,2,3,0.04)
    corners=cv2.dilate(corners,None)
    
    imgROI[corners>0.1*corners.max()]=[255,0,0]
    #print imgpoints[1][2][0]-imgpoints[0][2][0]
    plt.imshow(img)
    plt.show()
                 
#cv2.destroyAllWindows()