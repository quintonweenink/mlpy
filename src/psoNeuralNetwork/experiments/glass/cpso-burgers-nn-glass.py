from psoNeuralNetwork.psonn import PSONN

from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = PSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = dataSetTool.getGlassDataSets('../../../dataSet/glass/glass.data')

psonn.bounds = Bounds(-5, 5)

# Create neural network
psonn.createNeuralNetwork([12])

# Create the pso with the nn weights
psonn.num_particles = 40
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 1.4
psonn.socialConstant = 1.4

from numberGenerator.chaos.burgers import Burgers
psonn.numberGenerator = Burgers()

psonn.train()