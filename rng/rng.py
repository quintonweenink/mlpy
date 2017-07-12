import six
import abc

class RNG:
    __name = None
    __sound = ""
    __owner = ""

    def __init__(self, name, sound, owner):
        self.__name = name
        self.__sound = sound
        self.__owner = owner

    def getRandomSet(self):
        return self.__name

    def toString(self):
        pass