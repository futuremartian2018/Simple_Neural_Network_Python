# FILE THAT CONTAINS ACTIVATION FUNCTIONS

import numpy as np


# SIGMOID ACTIVATION FUNCTION
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# HYPERBOLIC TANGENT FUNCTION
def tanh(x):
    return (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))


# RECTIFIED LINEAR UNITS - ReLu
def relu(x, y = 0):
    m = np.array([x, y])
    return np.max(m)
