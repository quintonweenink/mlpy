from PSONeuralNetwork.psonn import PSONN

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds

psonn = PSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = dataSetTool.getIrisDataSets('../dataSet/iris/iris.data')

psonn.bounds = Bounds(-10, 10)

# Create neural network
l_rate = None
inputLayer = Layer(psonn.bounds, size = 4, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(psonn.bounds, size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(psonn.bounds, size = 3, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")
psonn.nn = NeuralNetwork()
psonn.nn.appendLayer(inputLayer)
psonn.nn.appendLayer(hiddenLayer)
psonn.nn.appendLayer(outputLayer)

# Create the pso with the nn weights
psonn.num_particles = 30
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.49445
psonn.socialConstant = 1.49445

psonn.train()