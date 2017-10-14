import numpy as np
import math

def gradientDecent(target):
    return 0.5 * sum((target - self.result) ** 2)

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def hyperTan(x, deriv=False):
    if x < -20.0:
        return -1.0
    elif x > 20.0:
        return 1.0
    else:
        return x #math.Tanh(x)