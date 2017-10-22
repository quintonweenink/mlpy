from psoNeuralNetwork.psonn import PSONN

from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = PSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.generalization, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

psonn.bounds = Bounds(-5, 5)

# Create neural network
psonn.createNeuralNetwork([8])

# Create the pso with the nn weights
psonn.num_particles = 30
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.49445
psonn.socialConstant = 1.49445

psonn.vmax = 0.1

from numberGenerator.chaos.dissipative import Dissipative
psonn.numberGenerator = Dissipative()

psonn.train()