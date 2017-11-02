import math
import numpy as np

from numberGenerator.chaos.cprng import CPRNG

class Burgers(CPRNG):

    def __init__(self, A = 0.75, B = 1.75):
        self._A = A
        self._B = B
        super(Burgers, self).__init__()


    def getNext(self):
        xn = self._A * self.x - math.pow(self.y, 2)
        yn = self._B * self.y + self.x * self.y

        self.x = xn
        self.y = yn

        return np.array([self.x, self.y])