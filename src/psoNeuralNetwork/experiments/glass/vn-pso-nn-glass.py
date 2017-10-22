import numpy as np

from psoNeuralNetwork.vonNeumannPSONN import VNPSONN
from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds


color = 'blue'

print('Random')
for _ in range(2):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getGlassDataSets('../../../dataSet/glass/glass.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([12])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = 0.01

    psonn.color = 'red'
    psonn.train()

print('Lozi: ')
for _ in range(2):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getGlassDataSets('../../../dataSet/glass/glass.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([12])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = 0.01

    from numberGenerator.chaos.lozi import Lozi
    psonn.numberGenerator = Lozi()

    psonn.color = 'green'
    psonn.train()

print('Dissipative: ')
for _ in range(2):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getGlassDataSets('../../../dataSet/glass/glass.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([12])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = 0.01

    from numberGenerator.chaos.dissipative import Dissipative
    psonn.numberGenerator = Dissipative()

    psonn.color = 'blue'
    psonn.train()