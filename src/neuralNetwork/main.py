import numpy as np

from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.layer import Layer

l_rate = 1.0

inputLayer = Layer(size = 2, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 4, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 1, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

input = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])

target = np.array([[0],
                   [1],
                   [1],
                   [0]])

for i in range(60000):
    mod = i % 4
    i_input = np.array([input[i % 4]])
    fnn.fire(i_input)
    i_target = np.array([target[i % 4]])
    fnn.backPropagation(i_target)

    if (i % 10000) == 0:
        print("Error:" + str(fnn))

print("FIRE: " + str(fnn.fire(np.array([input[0]]))))
print("FIRE: " + str(fnn.fire(np.array([input[1]]))))
print("FIRE: " + str(fnn.fire(np.array([input[2]]))))
print("FIRE: " + str(fnn.fire(np.array([input[3]]))))
