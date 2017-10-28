import numpy as np
import matplotlib.pyplot as plt

from psoNeuralNetwork.vonNeumannPSONN import VNPSONN
from dataSetTool import DataSetTool
from numberGenerator.bounds import Bounds

cpso_tinkerbell_errors = []
cpso_tinkerbell_generalization_error = []

cpso_lozi_errors = []
cpso_lozi_generalization_error = []

cpso_dissipative_errors = []
cpso_dissipative_generalization_error = []

pso_errors = []
pso_generalization_error = []

iterations = 10

V_MAX = None

print('Tinkerbell:')
for _ in range(iterations):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([8])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = V_MAX

    from numberGenerator.chaos.tinkerbell import Tinkerbell
    psonn.numberGenerator = Tinkerbell()

    psonn.color = 'red'
    trainingErrors, generalizationError = psonn.train()

    cpso_tinkerbell_errors.append(trainingErrors)
    cpso_tinkerbell_generalization_error.append(generalizationError)

print('Lozi:')
for _ in range(iterations):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([8])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = V_MAX

    from numberGenerator.chaos.lozi import Lozi
    psonn.numberGenerator = Lozi()

    psonn.color = 'green'
    trainingErrors, generalizationError = psonn.train()

    cpso_lozi_errors.append(trainingErrors)
    cpso_lozi_generalization_error.append(generalizationError)

print('Dissipative:')
for _ in range(iterations):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([8])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = V_MAX

    from numberGenerator.chaos.dissipative import Dissipative
    psonn.numberGenerator = Dissipative()

    psonn.color = 'blue'
    trainingErrors, generalizationError = psonn.train()

    cpso_dissipative_errors.append(trainingErrors)
    cpso_dissipative_generalization_error.append(generalizationError)

print('Random:')
for _ in range(iterations):
    psonn = VNPSONN()

    # Get data set
    dataSetTool = DataSetTool()
    psonn.training, psonn.generalization, psonn.testing = dataSetTool.getIrisDataSets('../../../dataSet/iris/iris.data')

    psonn.bounds = Bounds(-5, 5)

    psonn.createNeuralNetwork([8])

    # Create the pso with the nn weights
    psonn.num_particles_x = 5
    psonn.num_particles_y = 5

    psonn.inertia_weight = 0.729844
    psonn.cognitiveConstant = 1.496180
    psonn.socialConstant = 1.496180

    psonn.vmax = V_MAX

    psonn.color = 'black'
    trainingErrors, generalizationError = psonn.train()

    pso_errors.append(trainingErrors)
    pso_generalization_error.append(generalizationError)

iterations = [y[1] for y in pso_errors[0]]

cpso_tinkerbell_errors_no_iteration = [[y[0] for y in x] for x in cpso_tinkerbell_errors]
cpso_tinkerbell_errors_mean = np.mean(cpso_tinkerbell_errors_no_iteration, axis=0)
cpso_tinkerbell_generalization_error_mean = np.mean(cpso_tinkerbell_generalization_error)
cpso_tinkerbell_generalization_error_std = np.std(cpso_tinkerbell_generalization_error)

cpso_lozi_errors_no_iteration = [[y[0] for y in x] for x in cpso_lozi_errors]
cpso_lozi_errors_mean = np.mean(cpso_lozi_errors_no_iteration, axis=0)
cpso_lozi_generalization_error_mean = np.mean(cpso_lozi_generalization_error)
cpso_lozi_generalization_error_std = np.std(cpso_lozi_generalization_error)

cpso_dissipative_errors_no_iteration = [[y[0] for y in x] for x in cpso_dissipative_errors]
cpso_dissipative_errors_mean = np.mean(cpso_dissipative_errors_no_iteration, axis=0)
cpso_dissipative_generalization_error_mean = np.mean(cpso_dissipative_generalization_error)
cpso_dissipative_generalization_error_std = np.std(cpso_dissipative_generalization_error)

pso_errors_no_iteration = [[y[0] for y in x] for x in pso_errors]
pso_errors_mean = np.mean(pso_errors_no_iteration, axis=0)
pso_generalization_error_mean = np.mean(pso_generalization_error)
pso_generalization_error_std = np.std(pso_generalization_error)

print('- Tinkerbell:')
print('Errors: ', cpso_tinkerbell_errors_mean)
print('Mean squared error: ', cpso_tinkerbell_generalization_error_mean)
print('Standard deviation: ', cpso_tinkerbell_generalization_error_std)
print('- Lozi:')
print('Errors: ', cpso_lozi_errors_mean)
print('Mean squared error: ', cpso_lozi_generalization_error_mean)
print('Standard deviation: ', cpso_lozi_generalization_error_std)
print('- Dissipative:')
print('Errors: ', cpso_dissipative_errors_mean)
print('Mean squared error: ', cpso_dissipative_generalization_error_mean)
print('Standard deviation: ', cpso_dissipative_generalization_error_std)
print('- Random:')
print('Errors: ', pso_errors_mean)
print('Mean squared error: ', pso_generalization_error_mean)
print('Standard deviation: ', pso_generalization_error_std)

plt.close()

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0, 1])
plt.xlim([0, 5000])
plt.ion()

tinkerbell, = plt.plot(iterations, cpso_tinkerbell_errors_mean, color='red')
lozi, = plt.plot(iterations, cpso_lozi_errors_mean, color='green')
dissipative, = plt.plot(iterations, cpso_dissipative_errors_mean, color='blue')
random, = plt.plot(iterations, pso_errors_mean, color='black')
plt.legend([tinkerbell, lozi, dissipative, random], ['Tinkerbell', 'Lozi', 'Dissipative', 'Random'])
plt.show(5)