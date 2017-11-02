import matplotlib.pyplot as plt
import numpy as np

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

np.set_printoptions(suppress=True)

dataSetTool = DataSetTool()
training, generalization, testing = dataSetTool.getHeartDataSets('../../dataSet/heart/processed.cleveland.data')

plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ion()

l_rate = 0.1

bounds = Bounds(-2, 2)

inputLayer = Layer(bounds, size = len(training[0][0]), prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(bounds, size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(bounds, size = len(training[0][1]), prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

group_training = np.array([input[0] for input in training])
group_target = np.array([output[1] for output in training])

errors = []

for i in range(8000):
    mod = i % len(training)
    in_out = training[mod]
    result = fnn.fire(group_training)
    i_error = fnn.backPropagation(group_target)

    #print("Error:" + str(fnn))
    error = fnn.getMSE()
    errors.append(error)
    print(error)
    if i % 53 == 0:
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

correct = 0

for i in range(len(testing)):
    in_out = testing[i]
    result = fnn.fire(np.array([in_out[0]]))

    print(result)
    print(in_out[1])
    print()

    if np.argmax(result) == np.argmax(in_out[1]):
        correct += 1


print("Classification error: ", str(correct/len(testing)) + "%")

