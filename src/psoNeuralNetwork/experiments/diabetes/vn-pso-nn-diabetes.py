import numpy as np

from psoNeuralNetwork.vonNeumannPSONN import VNPSONN
from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = VNPSONN()

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

psonn.vmax = 0.1

psonn.train()