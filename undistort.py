import numpy as np
import cv2
import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
i=0

os.chdir("/Users/eren/Desktop/ComputerVision/MotionTracking")
f = open('dir.txt', 'r')
lines = f.readlines()
os.chdir(lines[0])

types = ('*.jpg', '*.JPG')
images = []
for files in types:
    images.extend(glob.glob(files))

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
    h,  w = img.shape[:2]
    print h
    print w
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(str(i) + '_2.jpg',dst)
    i=i+1
cv2.destroyAllWindows()