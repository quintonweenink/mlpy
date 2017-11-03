from numberGenerator.chaos.cprng import CPRNG

class Lozi(CPRNG):

    def __init__(self, A = 1.7, B = 0.5):
        self._A = A
        self._B = B
        super(Lozi, self).__init__()

    def getNext(self):
        xn = 1 - (self._A * abs(self.x)) + self.y
        yn = self._B * self.x

        self.x = xn
        self.y = yn

        return [self.x, self.y]