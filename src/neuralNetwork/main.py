import numpy as np

from neuralNetwork.feedForwardNeuralNetwork import FeedForwardNeuralNetwork
from neuralNetwork.layer import Layer

inputLayer = Layer(size = 2, prev = None, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 3, prev = inputLayer, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 1, prev = hiddenLayer, bias = False, label = "Output layer")

fnn = FeedForwardNeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

input = np.array([[1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 0]])

target = np.array([[0],
                   [1],
                   [1],
                   [0]])

output = fnn.fire(input[0])
print(fnn)

output = fnn.backPropagation(target[0])

print(fnn)

for i in range(10):
    mod = i % 4
    fnn.fire(input[mod])
    fnn.backPropagation(target[mod])

print("FIRE: " + str(fnn.fire(input[1])))