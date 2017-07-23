import random

from numberGenerator import NG

class CPRNG(NG):

    def __init__(self):
        pass

    def getRandomSet(self, x, y):
        return random.uniform(-1, 1)

    def random(self):
        return random.random()

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"