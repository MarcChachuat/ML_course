# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    gram=np.dot(np.transpose(tx),tx)
    w=np.dot(np.dot(np.linalg.inv(gram),np.transpose(tx)),y)
    e=y-np.dot(tx,w)
    mse=np.linalg.norm(e)**2
    mse=mse/(2*len(y))
    return mse,w
