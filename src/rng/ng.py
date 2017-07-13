import six
import abc

@six.add_metaclass(abc.ABCMeta)
class NG:
    __name = None

    def __init__(self, name):
        self.__name = name

    @abc.abstractmethod
    def getRandomSet(self):
        return self.__name

    @abc.abstractmethod
    def toString(self):
        pass

    @abc.abstractmethod
    def random(self):
        pass