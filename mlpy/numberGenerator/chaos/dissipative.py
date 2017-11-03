import math

from numberGenerator.chaos.cprng import CPRNG

class Dissipative(CPRNG):

    def __init__(self, B = 0.6, k = 8.8):
        self._B = B
        self._k = k
        super(Dissipative, self).__init__()

    def getNext(self):
        yn = (self._B * self.y + self._k * math.sin(self.x)) % (2 * math.pi)
        if yn < 0:
            yn += 2 * math.pi
        xn = (self.x + yn) % (2 * math.pi)
        if xn < 0:
            xn += 2 * math.pi

        self.x = xn
        self.y = yn

        return [self.x, self.y]