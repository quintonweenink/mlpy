import random

from experiments import math
from numberGenerator.chaos.cprng import CPRNG


class Tinkerbell(CPRNG):

    def __init__(self, a = 0.9, b = -0.6, c = 2, d = 0.5):
        self._A = a
        self._B = b
        self._C = c
        self._D = d
        super(Tinkerbell, self).__init__()


    def getRandomSet(self, x, y):
        pass

    def random(self):
        xn = math.pow(self._x, 2) - math.pow(self._y, 2) + (self._A * self._x) + (self._B * self._y)
        yn = (2 * self._x * self._y) + (self._C * self._x) + (self._D * self._y)

        self.x = xn
        self.y = yn

        return (self.x, self.y)

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"