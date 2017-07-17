import random

from numberGenerator.ng import NG

class RNG(NG):

    def __init__(self):
        pass

    def getRandomSet(self):
        return random.uniform(-1, 1)

    def random(self):
        return random.random()

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"