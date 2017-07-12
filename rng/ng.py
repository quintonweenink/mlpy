import six
import abc

@six.add_metaclass(abc.ABCMeta)
class NG:
    __name = None
    __sound = ""
    __owner = ""

    def __init__(self, name, sound, owner):
        self.__name = name
        self.__sound = sound
        self.__owner = owner

    @abc.abstractmethod
    def getRandomSet(self):
        return self.__name

    @abc.abstractmethod
    def toString(self):
        pass