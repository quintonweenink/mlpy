import random
import abc
import six
import numpy as np

from mlpy.numberGenerator.ng import NG

@six.add_metaclass(abc.ABCMeta)
class CPRNG(NG):

    def __init__(self):
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0, 1)
        self.reevaluation = 0
        self.listLen = 5000
        self.chaoticList = np.zeros((self.listLen, 2))

        self.pos = 0

        self.generateChaoticData()

    def generateChaoticData(self):
        self.chaoticList[0] = np.array([self.x, self.y])

        self.reevaluation += 1

        index = 1
        while index < len(self.chaoticList):
            self.chaoticList[index] = self.getNext()
            index += 1

        # Regenerate if list has inf or nan in it
        if np.isnan(self.chaoticList).any() == True or np.isinf(self.chaoticList).any() == True:
            self.x = random.uniform(0, 1)
            self.y = random.uniform(0, 1)
            self.generateChaoticData()

        max = np.max(self.chaoticList)
        min = np.min(self.chaoticList)

        # Normalization per listLen size
        self.chaoticList = (self.chaoticList - min) / (max - min)


    def random(self):
        if self.pos < self.listLen - 1:
            self.pos += 1
            return self.chaoticList[self.pos]
        else:
            self.generateChaoticData()
            self.pos = 0
            return self.chaoticList[self.pos]

    def randomArray(self, size):
        return np.array([self.random()[0] for i in range(size)])

    @abc.abstractmethod
    def getNext(self):
        pass