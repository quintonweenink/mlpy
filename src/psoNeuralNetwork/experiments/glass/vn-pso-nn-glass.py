import numpy as np
import matplotlib.pyplot as plt

from psoNeuralNetwork.vonNeumannPSONN import VNPSONN
from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds


color = 'blue'

print('Random')

pso_errors = []
pso_generalization_error = []

cpso_lozi_errors = []
cpso_lozi_generalization_error = []

cpso_dissipative_errors = []
cpso_dissipative_generalization_error = []

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
    trainingErrors, generalizationError = psonn.train()

    pso_errors.append(trainingErrors)
    pso_generalization_error.append(generalizationError)

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
    trainingErrors, generalizationError = psonn.train()

    cpso_lozi_errors.append(trainingErrors)
    cpso_lozi_generalization_error.append(generalizationError)

print('Dissipative: ')
for _ in range(0):
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
    trainingErrors, generalizationError = psonn.train()

    cpso_dissipative_errors.append(trainingErrors)
    cpso_dissipative_generalization_error.append(generalizationError)

iterations = [y[1] for y in pso_errors[0]]

pso_errors_no_iteration = [[y[0] for y in x] for x in pso_errors]
pso_errors_mean = np.mean(pso_errors_no_iteration, axis=0)
pso_generalization_error_mean = np.mean(pso_generalization_error)

cpso_lozi_errors_no_iteration = [[y[0] for y in x] for x in cpso_lozi_errors]
cpso_lozi_errors_mean = np.mean(cpso_lozi_errors_no_iteration, axis=0)
cpso_lozi_generalization_error_mean = np.mean(cpso_lozi_generalization_error)

print(pso_generalization_error_mean)
print(cpso_lozi_generalization_error_mean)


plt.close()

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0, 1])
plt.xlim([0, 5000])
plt.ion()

random, = plt.plot(iterations, pso_errors_mean, color='red')
lozi, = plt.plot(iterations, cpso_lozi_errors_mean, color='green')
plt.legend([random, lozi], ['Random', 'Lozi'])
plt.show(5)