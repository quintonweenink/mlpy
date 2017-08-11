import numpy as np

def nonlin(x, deriv = False):
    if(deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))
#
# def addBias(vec):
#     return np.insert(vec, len(vec[0]), 1.0, axis=1)
#
# # input data
# x = np.array([[0,0],
#               [0,1],
#               [1,0],
#               [1,1]])
#
# # output data
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])
#
# np.random.seed(1)
#
# syn0 = (2 * np.random.random((3,5))) - 1
# print(syn0)
# syn1 = (2 * np.random.random((5,1))) - 1
# print(syn1)
#
# # trainig step
# for i in range(1):
#     l0 = addBias(x)
#     print(l0)
#     l1 = nonlin(np.dot(l0, syn0))
#     print(l1)
#     l1 = addBias(l1)
#     print(l1)
#     l2 = nonlin(np.dot(addBias(l1), syn1))
#     print(l2)
#
#     l2_error = y - l2
#
#     if(i % 1000) == 0:
#         print("Error:" + str(np.mean(np.abs(l2_error))))
#
#     l2_delta = l2_error * nonlin(l2, deriv=True)
#
#     l1_error = l2_delta.dot(syn1.T)
#
#     l1_delta = l1_error * nonlin(l1, deriv=True)
#
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += l0.T.dot(l1_delta)
#
# print("Output after training")
#
# print(l2)

class FeedForwardNeuralNetwork(object):
    def __init__(self):
        self._layers = []

    def appendLayer(self, layer):
        assert isinstance(layer, Layer)

        self._layers.append(layer)

    def canFire(self):
        return len(self._layers) > 1

    def fire(self, input):
        assert len(input) == self._layers[0].size
        assert self.canFire()

        self._layers[0].delta = input

        for layer in self._layers:
            layer.propagate()

        return self._layers[len(self._layers) - 1].delta




class Layer(object):
    def __init__(self, size, prev, bias = False, label = "Layer"):
        assert isinstance(size, int) and size > 0
        assert isinstance(prev, Layer) or prev == None
        assert isinstance(bias, bool)
        assert isinstance(label, str)

        self.label = label

        self.prev = prev
        self.size = size
        self.syn = None

        self.delta = []

        if not prev == None:
            prev.syn = (2 * np.random.random((prev.size, self.size))) - 1
            print(prev.syn)

    def toString(self):
        print(self.syn)

    def propagate(self):
        if self.prev != None:
            for i in range(self.size):
                self.delta.append(np.mean(self.prev.syn[0]))


inputLayer = Layer(size = 2, prev = None, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 4, prev = inputLayer, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 1, prev = hiddenLayer, bias = False, label = "Output layer")

fnn = FeedForwardNeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

print(fnn.fire([1,1]))






