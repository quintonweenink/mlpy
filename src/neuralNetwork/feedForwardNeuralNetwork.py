import numpy as np

from neuralNetwork.layer import Layer

class FeedForwardNeuralNetwork(object):
    def __init__(self):
        self._layers = []

    def appendLayer(self, layer):
        assert isinstance(layer, Layer)

        self._layers.append(layer)

    def canFire(self):
        return len(self._layers) > 1

    def fire(self, input):
        assert isinstance(input, (np.ndarray, np.generic))
        assert len(input) == self._layers[0].size
        assert self.canFire()

        self._layers[0].result = input

        for layer in self._layers:
            layer.propagate()

        return self._layers[len(self._layers) - 1].result

    def backPropagation(self, target):
        for layer in reversed(self._layers):
            target = layer.backPropagate(target)
            print(layer)

        for layer in self._layers:
            layer.applyDeltaWeights()

        return self._layers[len(self._layers) - 1].result

    def __str__(self):
        res = ""
        for layer in self._layers:
            res += str(layer)
        return res