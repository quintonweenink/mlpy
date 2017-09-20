import numpy as np

from psoNeuralNetwork.psonn import PSONN
from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

psonn = PSONN()

# Get data set
dataSetTool = DataSetTool()
psonn.training, psonn.testing = dataSetTool.getGlassDataSets('../../../dataSet/glass/glass.data')

psonn.bounds = Bounds(-10, 10)

psonn.createNeuralNetwork([12])

# Create the pso with the nn weights
psonn.num_particles = 40
psonn.inertia_weight = 0.729
psonn.cognitiveConstant = 0.9
psonn.socialConstant = 0.9

psonn.train()

errors = []
psonn.nn.setAllWeights(psonn.pso.group_best_position)
for i in range(len(psonn.testing)):
    in_out = psonn.testing[i]
    result = psonn.nn.fire(np.array([in_out[0]]))

    print(result)
    print(in_out[1])
    print()