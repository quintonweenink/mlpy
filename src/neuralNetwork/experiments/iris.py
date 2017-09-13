import matplotlib.pyplot as plt
import numpy as np

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

dataSetTool = DataSetTool()
training, testing = dataSetTool.getIrisDataSets('../dataSet/iris/iris.data')

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,1])
plt.ion()

l_rate = 0.5

bounds = Bounds(-2, 2)

inputLayer = Layer(bounds, size = 4, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(bounds, size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(bounds, size = 3, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

errors = []

for i in range(8000):
    mod = i % len(training)
    in_out = training[mod]
    fnn.fire(np.array([in_out[0]]))
    i_error = fnn.backPropagation(np.array([in_out[1]]))

    if (i % 35) == 0:
        #print("Error:" + str(fnn))
        errors.append(abs(i_error[0][0]))
        plt.scatter(i, abs(i_error[0][0]), color='blue', s=4, label="test1")
        plt.pause(0.0001)
        plt.show()


plt.pause(5)
print(fnn)

plt.close()

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,1])
plt.ion()

for i in range(len(testing)):
    mod = i % len(testing)
    in_out = testing[mod]
    fnn.fire(np.array([in_out[0]]))
    i_error = fnn.backPropagation(np.array([in_out[1]]))

    errors.append(abs(i_error[0][0]))
    plt.scatter(i, abs(i_error[0][0]), color='blue', s=4, label="test1")
    plt.pause(0.01)
    plt.show()

plt.pause(5)
print(fnn)

weights = fnn.getAllWeights()
fnn.setAllWeights(weights)


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

