import random
import abc
import six

from numberGenerator.ng import NG
from numberGenerator.rng import RNG

@six.add_metaclass(abc.ABCMeta)
class CPRNG(NG):

    def __init__(self):
        self.__rng = RNG()
        self.__x = self.__rng.random()
        self.__y = self.__rng.random()

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

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value