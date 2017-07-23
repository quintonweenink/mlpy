import random
import math

from numberGenerator import CPRNG

class Lozi(CPRNG):

    def __init__(self, a = 1.7, b = 0.5):
        self.__a = a
        self.__b = b

    def getRandomSet(self, x, y):
        yn = x
        xn = 1 - self.__a * abs(x) + self.__b * y

        return (xn, yn)

    def random(self):
        return random.random()

    def uniform(self, x, y):
        return random.uniform(x, y)

    def toString(self):
        return "Printing the set"