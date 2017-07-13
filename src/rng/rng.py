from ng import NG

class RNG(NG):

    def __init__(self, name):
        super(RNG, self).__init__(name)

    def getRandomSet(self):
        return self.__name

    def toString(self):
        return "Printing the set"