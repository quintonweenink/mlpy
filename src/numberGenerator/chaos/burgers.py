import random

from experiments import math
from numberGenerator.chaos.cprng import CPRNG


class Burgers(CPRNG):

    def __init__(self, a = 1.7, b = 0.5):
        self._A = a
        self._B = b
        super(Burgers, self).__init__()


    def getRandomSet(self, x, y):
        pass

    def random(self):
        xn = self._A * self._x - math.pow(self._y, 2)
        yn = self._B * self._y + self._x * self.y

        self.x = xn
        self.y = yn

        return (self.x, self.y)

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"