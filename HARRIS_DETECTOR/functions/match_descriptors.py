import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    distances = np.zeros([desc1.shape[0],desc2.shape[0]])
    
    for i, val1 in enumerate(desc1):
        for j, val2 in enumerate(desc2):
            diff = val1-val2
            distances[i,j] = np.linalg.norm(diff)
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        index1 = np.arange(q1)
        minMatch = np.argmin(distances, axis=1)
        matches = np.column_stack((index1, minMatch))       
        return matches
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        minMatch1 = np.argmin(distances, axis=1)
        minMatch2 = np.argmin(distances, axis=0)
        matches1 = np.column_stack((np.arange(q1), minMatch1))
        matches2 = np.column_stack((minMatch2, np.arange(q2)))
        set1 = set([tuple(x) for x in matches1])
        set2 = set([tuple(x) for x in matches2])           
        matches = np.array([x for x in set1&set2])
        
        return matches
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        minMatch1 = np.argmin(distances, axis=1)
        minMatch2 = np.partition(distances,2,axis=0)[:,1]
        r = minMatch1/minMatch2
        matches_set = set([tuple(x) for x in np.column_stack((np.arange(q1), minMatch1))])
        matches = np.array([ele for i, ele in enumerate(matches_set) if r[i] > 50])
        return matches
    else:
        raise NotImplementedError


