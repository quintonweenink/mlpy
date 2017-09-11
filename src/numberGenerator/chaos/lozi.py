import random

from numberGenerator.chaos.cprng import CPRNG

class Lozi(CPRNG):

    def __init__(self, a = 1.7, b = 0.5):
        self._A = a
        self._B = b
        super(Lozi, self).__init__()


    def getRandomSet(self, x, y):
        pass

    def random(self):
        xn = 1 - (self._A * abs(self.x)) + self.y
        yn = self._B * self.x

        self.x = xn
        self.y = yn

        return (self.x, self.y)

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"