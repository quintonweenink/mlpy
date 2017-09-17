import random

from numberGenerator.ng import NG

class RNG(NG):

    def __init__(self):
        pass

    def random(self):
        return random.random()