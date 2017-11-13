import numpy as np
import cv2
cap = cv2.VideoCapture('unstable.avi')

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(100, 100), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Take first frame and find corners in it
ret, old_frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
rows, cols, c = old_frame.shape

good_old = p0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stable_cv2.avi', fourcc, fps, (cols, rows))

while(1):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st == 1]
    
    M = cv2.getAffineTransform(good_new[0:3], good_old[0:3])
    img = cv2.warpAffine(frame, M, (cols, rows))

    out.write(img)
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()
