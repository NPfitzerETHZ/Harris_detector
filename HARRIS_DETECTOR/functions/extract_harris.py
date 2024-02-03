import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter


import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 2e-4):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx = signal.convolve2d(img,Sx,mode='same')
    Gy = signal.convolve2d(img,Sy,mode='same')
    
    # 2. Blur the computed gradients
    Ixx = cv2.GaussianBlur(Gx**2,[0,0],sigma,cv2.BORDER_REPLICATE)
    Iyy = cv2.GaussianBlur(Gy**2,[0,0],sigma,cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Gx*Gy,[0,0],sigma,cv2.BORDER_REPLICATE)
    
    # 3. Compute elements of the local auto-correlation matrix "M" 
    det = Ixx*Iyy-Ixy**2
    trace = Ixx**2+Iyy**2
    R = det - k*trace**2
    # print(R)

    # 4. Compute Harris response function C 
    # 5. Detection with threshold and non-maximum suppression
    C = R
    L = ndimage.maximum_filter(R,(20,20))
    L = np.where(L == R, L, 0)
    L = np.where(L > thresh, 1, 0)

    cx,cy = np.nonzero(L)
    corners = np.zeros([len(cx),2])
    corners[:,0] = cy
    corners[:,1] = cx

    return corners, C

