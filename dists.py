import numpy as np

def manhattan(p1, p2):
    """
    Returns the norm 1 between two elements of the class shapely.Point
    """
    return np.abs(p1.x - p2.x) + np.abs(p1.y - p2.y)

def dist(p1, p2):
    """
    Returns the norm 2 between two elements of the class shapely.Point
    """
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)