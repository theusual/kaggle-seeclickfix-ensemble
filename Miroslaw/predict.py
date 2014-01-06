import numpy as np
from matplotlib.mlab import find
from data_transforms import create_weight 

def apply_bound(pred, bounds=(0,1,0)):
    """
    This ensures that all views and comments >= 0 and all votes >= 1
    """
    pred = (pred >= bounds) * pred + bounds * (pred < bounds)
    return pred

def apply_scales(pred, categories, scales):
    """
    Applies scales to a prediction given a dict containing scales indexed
    by category name and a list of categories
    
    len(categories) == pred.shape[0]
    """
    weights = create_weight(categories, scales)
    return apply_bound(pred * weights)
