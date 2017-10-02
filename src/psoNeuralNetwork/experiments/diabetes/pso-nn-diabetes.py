import numpy as np

from psoNeuralNetwork.psonn import PSONN
from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = PSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = dataSetTool.getPrimaIndiansDiabetesSets('../../../dataSet/pima-indians-diabetes/pima-indians-diabetes.data')

psonn.bounds = Bounds(-5, 5)

psonn.createNeuralNetwork([12])

# Create the pso with the nn weights
psonn.num_particles = 40
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.4
psonn.socialConstant = 0.6

psonn.train()