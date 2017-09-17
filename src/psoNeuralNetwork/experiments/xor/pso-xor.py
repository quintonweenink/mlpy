import numpy as np

from psoNeuralNetwork.psonn import PSONN
from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

psonn = PSONN()

input = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])

target = np.array([[0],
                   [1],
                   [1],
                   [0]])

training = []
for x, y in zip(input, target):
    training.append((x, y))

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = training, training

psonn.bounds = Bounds(-5, 5)

# Create neural network
l_rate = None
inputLayer = Layer(psonn.bounds, size = 2, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(psonn.bounds, size = 4, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(psonn.bounds, size = 1, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")
psonn.nn = NeuralNetwork()
psonn.nn.appendLayer(inputLayer)
psonn.nn.appendLayer(hiddenLayer)
psonn.nn.appendLayer(outputLayer)

# Create the pso with the nn weights
psonn.num_particles = 20
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.49445
psonn.socialConstant = 1.49445

psonn.train()

print("FIRE: " + str(psonn.nn.fire(np.array([input[0]]))))
print("FIRE: " + str(psonn.nn.fire(np.array([input[1]]))))
print("FIRE: " + str(psonn.nn.fire(np.array([input[2]]))))
print("FIRE: " + str(psonn.nn.fire(np.array([input[3]]))))
