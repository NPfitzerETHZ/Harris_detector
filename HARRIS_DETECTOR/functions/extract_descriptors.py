import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    '''
    Inputs:
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    Returns:
    - keypoints:    (q', 2) numpy array of keypoint locations [x, y] that are far enough from edges
    '''
    testkeys_X = keypoints[:,0]
    testkeys_Y = keypoints[:,1]
    newkeys_X = [ele for ele in testkeys_X if (ele >= patch_size//2 and ele <= np.shape(img)[1]-patch_size//2)]
    newkeys_Y = [testkeys_Y[i] for i, ele in enumerate(testkeys_X) if (ele >= patch_size//2 and ele <= np.shape(img)[1]-patch_size//2)]

    newkeys_YY = np.array([ele for ele in newkeys_Y if (ele >= patch_size//2 and ele <= np.shape(img)[0]-patch_size//2)])
    newkeys_XX = np.array([newkeys_X[i] for i, ele in enumerate(newkeys_Y) if (ele >= patch_size//2 and ele <= np.shape(img)[0]-patch_size//2)])

    newkeys = np.zeros([len(newkeys_XX),2],dtype=int)
    newkeys[:,0] = newkeys_XX.astype(int)
    newkeys[:,1] = newkeys_YY.astype(int)
    
    return newkeys


def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

