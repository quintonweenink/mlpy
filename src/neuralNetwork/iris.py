import random

import numpy as np
import matplotlib.pyplot as plt

from feedForwardNeuralNetwork import NeuralNetwork
from layer import Layer

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,1])
plt.legend(['error'])
plt.ion()

l_rate = 0.5

inputLayer = Layer(size = 4, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 3, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

from numpy import genfromtxt
input = np.array(genfromtxt('dataSet/iris.data', delimiter=',', usecols=(0, 1, 2, 3)))
outputClassification = np.array(genfromtxt('dataSet/iris.data', dtype=str, delimiter=',', usecols=4))

classifications = []

for item in outputClassification:
    notInList = True
    for classification in classifications:
        if(classification == item):
            notInList = False
    if notInList:
        classifications.append(item)


output = []

for item in outputClassification:
    target = []
    for classification in classifications:
        if item == classification:
            target.append(1)
        else:
            target.append(0)
    output.append(target)

output = np.array(output)

input_target = []

for i in range(len(input)):
    input_target.append((input[i], output[i]))

random.shuffle(input_target)

training = input_target[:int(len(input_target)/2)]
testing = input_target[int(len(input_target)/2):]

errors = []

for i in range(8000):
    mod = i % len(training)
    in_out = training[mod]
    fnn.fire(np.array([in_out[0]]))
    i_error = fnn.backPropagation(np.array([in_out[1]]))

    if (i % 50) == 0:
        #print("Error:" + str(fnn))
        errors.append(abs(i_error[0][0]))
        plt.scatter(i, abs(i_error[0][0]))
        plt.pause(0.001)
        plt.draw()


plt.pause(5)
print(fnn)

plt.close()

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,1])
plt.legend(['error'])
plt.ion()

setosa = np.array([[5.0,3.3,1.4,0.2]])#Iris-setosa
setosa_o = np.array([[1,0,0]])
versicolor = np.array([[5.7,2.8,4.1,1.3]])#Iris-versicolor
versicolor_o = np.array([[0,1,0]])
virginica = np.array([[5.9,3.0,5.1,1.8]])#Iris-virginica
virginica_o = np.array([[0,0,1]])

print("FIRE: " + str(fnn.fire(setosa)))
print("OUT: " + str(setosa_o))
print("FIRE: " + str(fnn.fire(versicolor)))
print("OUT: " + str(versicolor_o))
print("FIRE: " + str(fnn.fire(virginica)))
print("OUT: " + str(virginica_o))

for i in range(len(testing)):
    mod = i % len(testing)
    in_out = testing[mod]
    fnn.fire(np.array([in_out[0]]))
    i_error = fnn.backPropagation(np.array([in_out[1]]))

    errors.append(abs(i_error[0][0]))
    plt.scatter(i, abs(i_error[0][0]))
    plt.pause(0.1)
    plt.draw()

plt.pause(5)
print(fnn)
