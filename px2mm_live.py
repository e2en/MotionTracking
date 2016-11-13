import numpy as np
import cv2
import undistort

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap=cv2.VideoCapture(0)

directory="/Users/eren/Desktop/ComputerVision/MotionTracking"
undistort.chdirectory(directory)

types = ('*.jpg', '*.JPG')
images=undistort.rddirectory(types)

mtx, dist=undistort.rdCalibrationResults()

while (True):
    
    ret1,img=cap.read()
    h,  w = img.shape[:2]
    gray=undistort.undistortion(img,h,w)
    cv2.imshow("gray", gray)
    
    ret, corners = cv2.findChessboardCorners(gray, (6,6),None)
    
    if ret==True: #chessboard pattern
        
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print corners[0][0][0]-corners[1][0][0]
            
    else: #corner pattern
        
        print "chessboard couldnt recognized!"

    if cv2.waitKey(100) & 0xFF==ord('q'):
        break

cap.release()    
cv2.destroyAllWindows()