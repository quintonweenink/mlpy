import matplotlib.pyplot as plt
import numpy as np

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

np.set_printoptions(suppress=True)

dataSetTool = DataSetTool()
fileName = '../../dataSet/pima-indians-diabetes/pima-indians-diabetes.data'
training, testing = dataSetTool.getPrimaIndiansDiabetesSets(fileName)

plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ion()

l_rate = 1

bounds = Bounds(-5, 5)

inputLayer = Layer(bounds, size = len(training[0][0]), prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(bounds, size = 20, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(bounds, size = len(training[0][1]), prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

group_training = np.array([input[0] for input in training])
group_target = np.array([output[1] for output in training])

errors = []

for i in range(400):
    mod = i % len(training)
    in_out = training[mod]
    result = fnn.fire(group_training)
    i_error = fnn.backPropagation(group_target)

    #print("Error:" + str(fnn))
    error = np.mean(np.square(fnn.layers[len(fnn.layers) - 1].error))
    errors.append(error)
    print(error)
    #print(fnn.layers[len(fnn.layers)-2].deltaWeights)
    #if i % 53 == 0:
    plt.scatter(i, abs(error), color='blue', s=4, label="test1")
    plt.pause(0.0001)
    plt.show()


plt.pause(5)

plt.close()

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,1])
plt.ion()


for i in range(len(testing)):
    in_out = testing[i]
    result = fnn.fire(np.array([in_out[0]]))

    print(result)
    print(in_out[1])
    print()

