import random
import numpy as np
import math

from numberGenerator.chaos.cprng import CPRNG


class Tinkerbell(CPRNG):

    def __init__(self, a = 0.9, b = -0.6, c = 2.0, d = 0.5):
        self._A = a
        self._B = b
        self._C = c
        self._D = d
        super(Tinkerbell, self).__init__()

    def getNext(self):
        xn = math.pow(self.x, 2) - math.pow(self.y, 2) + (self._A * self.x) + (self._B * self.y)
        yn = (2 * self.x * self.y) + (self._C * self.x) + (self._D * self.y)

        self.x = xn
        self.y = yn

        if xn < 1000 or xn > 1000 or yn < 1000 or yn > 1000:
            self.x = random.uniform(0, 1)
            self.y = random.uniform(0, 1)

        return np.array([self.x, self.y])