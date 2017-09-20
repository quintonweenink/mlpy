from psoNeuralNetwork.psonn import PSONN

from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = PSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

psonn.bounds = Bounds(-10, 10)

# Create neural network
psonn.createNeuralNetwork([8])

# Create the pso with the nn weights
psonn.num_particles = 30
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.49445
psonn.socialConstant = 1.49445

from numberGenerator.chaos.lozi import Lozi
psonn.numberGenerator = Lozi()

psonn.train()