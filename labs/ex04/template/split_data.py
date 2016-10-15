# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio: TODO
    indices=np.arange(x.shape[0])
    np.random.shuffle(indices)
    
    #upperbound for the ratio : 
    up=int(ratio * x.shape[0])
    #creation of the data sets
    xtrain=x[indices[0:up]]
    xtest=x[indices[up:]]
    ytrain=y[indices[0:up]]
    ytest=y[indices[up:]]
    #returning the result
    return xtrain,xtest,ytrain,ytest
