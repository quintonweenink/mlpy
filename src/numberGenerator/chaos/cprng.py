import random
import abc
import six

from numberGenerator.ng import NG
from numberGenerator.rng import RNG

@six.add_metaclass(abc.ABCMeta)
class CPRNG(NG):

    def __init__(self):
        self.__rng = RNG()
        self.x = self.__rng.random()
        self.y = self.__rng.random()

    @abc.abstractmethod
    def getRandomSet(self, x, y):
        pass

    @abc.abstractmethod
    def random(self):
        pass

    @abc.abstractmethod
    def uniform(self, x, y):
        pass

    def toString(self):
        return "Printing the set"