import matplotlib.pyplot as plt
import numpy as np

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

np.set_printoptions(suppress=True)

dataSetTool = DataSetTool()
training, generalization, testing = dataSetTool.getGlassDataSets('../../dataSet/glass/glass.data')

l_rate = 0.1

bounds = Bounds(-2, 2)

inputLayer = Layer(bounds, size = len(training[0][0]), prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(bounds, size = 12, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(bounds, size = len(training[0][1]), prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

group_training = np.array([input[0] for input in training])
group_target = np.array([output[1] for output in training])

errors = []

for i in range(5000):
    result = fnn.fire(group_training)
    i_error = fnn.backPropagation(group_target)

    #print("Error:" + str(fnn))
    difference = group_target - result
    fnn.error = np.mean(np.square(difference))

    if i % 100 == 0:
        errors.append(fnn.error)
        plt.scatter(i, abs(fnn.error), color='blue', s=4, label="test1")
        plt.pause(0.0001)
        plt.show()

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

