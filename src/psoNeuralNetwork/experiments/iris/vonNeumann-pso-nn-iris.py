from psoNeuralNetwork.vonNeumannPSONN import VNPSONN

from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = VNPSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

psonn.bounds = Bounds(-5, 5)

# Create neural network
psonn.createNeuralNetwork([8])

# Create the pso with the nn weights
psonn.num_particles = 8
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.49445
psonn.socialConstant = 0.69445

psonn.train()