import random

from ng import NG

class RNG(NG):

    def __init__(self, name):
        super(RNG, self).__init__(name)

    def getRandomSet(self):
        return self.__name

    def random(self):
        return random.random()

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"