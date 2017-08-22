import numpy as np
import matplotlib.pyplot as plt

from feedForwardNeuralNetwork import NeuralNetwork
from layer import Layer

def showSigmoidPlot():
    testValues = np.arange(-5, 5, 0.01)
    plt.plot(testValues, inputLayer.nonlin(testValues), linewidth=2)
    plt.plot(testValues, inputLayer.nonlin(inputLayer.nonlin(testValues), deriv=True), linewidth=2)
    plt.grid(1)
    plt.legend(['sigmoid', 'sigmoid derivative'])
    plt.show()

def plotIterationError(i, error):
    plt.scatter(i, error)

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend(['error'])
plt.ion()

l_rate = 0.1

inputLayer = Layer(size = 2, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 4, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
hiddenLayer2 = Layer(size = 4, prev = hiddenLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 1, prev = hiddenLayer2, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(hiddenLayer2)
fnn.appendLayer(outputLayer)

input = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])

target = np.array([[0],
                   [1],
                   [1],
                   [0]])

errors = []

for i in range(40000):
    mod = i % 4
    i_input = np.array([input[i % 4]])
    fnn.fire(i_input)
    i_target = np.array([target[i % 4]])
    i_error = fnn.backPropagation(i_target)

    if (i % 500) == 0:
        #print("Error:" + str(fnn))
        errors.append(abs(i_error[0][0]))
        plt.scatter(i, abs(i_error[0][0]))
        plt.show()
        plt.pause(0.0001)


plt.pause(5)

print("FIRE: " + str(fnn.fire(np.array([input[0]]))))
print("FIRE: " + str(fnn.fire(np.array([input[1]]))))
print("FIRE: " + str(fnn.fire(np.array([input[2]]))))
print("FIRE: " + str(fnn.fire(np.array([input[3]]))))
