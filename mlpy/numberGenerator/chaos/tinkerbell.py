import math
import random

import numpy as np

from mlpy.numberGenerator.chaos.cprng import CPRNG

class Tinkerbell(CPRNG):

    def __init__(self, a = 0.9, b = -0.6, c = 2.0, d = 0.5):
        self._A = a
        self._B = b
        self._C = c
        self._D = d

        self.x = -random.uniform(0.01, 0.1)
        self.y = random.uniform(0, 0.1)
        self.reevaluation = 0
        self.listLen = 5000
        self.chaoticList = np.zeros((self.listLen, 2))

        self.pos = 0

        self.generateChaoticData()

    def getNext(self):
        xn = math.pow(self.x, 2) - math.pow(self.y, 2) + (self._A * self.x) + (self._B * self.y)
        yn = (2 * self.x * self.y) + (self._C * self.x) + (self._D * self.y)

        self.x = xn
        self.y = yn

        return np.array([self.x, self.y])