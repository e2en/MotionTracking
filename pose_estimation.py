import cv2
import numpy as np
import glob
import os
import csv

###INITIALIZATIONS#####################################
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    #cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    #cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)*26

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

axis = np.float32([[0,0,0], [0,0,0], [0,0,-0]]).reshape(-1,3)
i=0
error=[]
first_point=[]
########################################################

os.chdir("/Users/eren/Desktop/ComputerVision/MotionTracking")
f = open('dir.txt', 'r')
lines = f.readlines()
os.chdir(lines[0])

types = ('*.jpg', '*.JPG')
images = []
for files in types:
    images.extend(glob.glob(files))
    
#os.chdir("/Users/eren/Desktop/ComputerVision/CalibrationImages")
###READ CALIBRATION MATRIX AND DISTORTION PARAMETERS###
f = open ( 'data.txt' , 'r')
l = []
l = [ line.split() for line in f]

l= map(float, l[0])+map(float, l[1])+map(float, l[2])+map(float, l[3])+map(float, l[4])+map(float, l[5])+map(float, l[6])+map(float, l[7])+map(float, l[8])+map(float, l[9]);
mtx=np.matrix([[l[0], l[1], l[2]],[0, l[3], l[4]],[0, 0, 1]])
dist=np.matrix([l[5], l[6], l[7], l[8], l[9]])
########################################################
    
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
    
    if ret == True:

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
        
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpoints.append(imgpts[0])
        print fname
        print tvecs
        
    img2 = draw(img,corners,imgpts)
    cv2.imshow('img',img2)

###COORDINATE TRANSFORMATIONS RELATIVE TO CAMERA AXIS###
    Rt=cv2.Rodrigues(rvecs)[0]
    R=np.matrix(Rt).T
    pos=-R*tvecs
    pos=pos.T
    #print tvecs
########################################################
 
cv2.waitKey(5)
cv2.destroyAllWindows()