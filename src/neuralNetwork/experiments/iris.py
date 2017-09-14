import matplotlib.pyplot as plt
import numpy as np

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

np.set_printoptions(suppress=True)

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

group_training = np.array([input[0] for input in training])
group_target = np.array([output[1] for output in training])

errors = []

for i in range(400):
    mod = i % len(training)
    in_out = training[mod]
    fnn.fire(group_training)
    i_error = fnn.backPropagation(group_target)

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
    result = fnn.fire(np.array([in_out[0]]))
    i_error = fnn.backPropagation(np.array([in_out[1]]))

    errors.append(abs(i_error[0][0]))
    plt.scatter(i, abs(i_error[0][0]), color='blue', s=4, label="test1")
    plt.pause(0.01)
    plt.show()
    print(result)
    print(in_out[1])
    print()

plt.pause(5)
#print(fnn)

