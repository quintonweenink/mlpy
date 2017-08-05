
class Bounds(object):


    def __init__(self, minBound = -1, maxBound = 1):
        self.__minBound = minBound
        self.__maxBound = maxBound

    @property
    def minBound(self):
        return self.__minBound

    @minBound.setter
    def minBound(self, value):
        self.__minBound = value

    @property
    def maxBound(self):
        return self.__maxBound

    @maxBound.setter
    def maxBound(self, value):
        self.__maxBound = value