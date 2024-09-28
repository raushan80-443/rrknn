import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def manhattan_distance(p1, p2):
    return np.sum(np.abs(np.array(p1) - np.array(p2)))
