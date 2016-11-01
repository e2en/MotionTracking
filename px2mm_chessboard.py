import numpy as np
import cv2
import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

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

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    #cv2.imshow('undistorted', dst)
    
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
    
    if ret == True:
        print (corners[0] - corners[1])
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print fname
        print (corners[0] - corners[1])
        
        for i in corners[1,:]:
            # draw the center of the circle
            cv2.circle(dst,(i[0],i[1]),2,(0,0,255),3)
            cv2.imshow('corners', dst)
            cv2.waitKey(1000)  
                 
cv2.destroyAllWindows()