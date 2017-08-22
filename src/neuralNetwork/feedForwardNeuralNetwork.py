import numpy as np

from layer import Layer

class NeuralNetwork(object):
    def __init__(self):
        self.layers = []

    def appendLayer(self, layer):
        assert isinstance(layer, Layer)

        self.layers.append(layer)

    def canFire(self):
        return len(self.layers) > 1

    def fire(self, input):
        assert isinstance(input, (np.ndarray, np.generic))
        assert len(input[0]) == self.layers[0].size
        assert self.canFire()

        self.layers[0].result = input

        for layer in self.layers:
            layer.propagate()

        return self.layers[len(self.layers) - 1].result

    def backPropagation(self, target):
        for layer in reversed(self.layers):
            target = layer.backPropagate(target)

        for layer in self.layers:
            layer.applyDeltaWeights()

        return self.layers[len(self.layers) - 1].error

    def __str__(self):
        res = ""
        for layer in self.layers:
            res += str(layer)
        return res