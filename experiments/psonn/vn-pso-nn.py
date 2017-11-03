import matplotlib.pyplot as plt
import numpy as np

from mlpy.dataSet.dataSetTool import DataSetTool
from mlpy.numberGenerator.bounds import Bounds
from mlpy.psoNeuralNetwork.vonNeumannPSONN import VNPSONN

dataSetTool = DataSetTool()

# Error arrays
cpso_tinkerbell_errors = []
cpso_tinkerbell_error = []
cpso_tinkerbell_generalization_error = []
cpso_lozi_errors = []
cpso_lozi_error = []
cpso_lozi_generalization_error = []
cpso_dissipative_errors = []
cpso_dissipative_error = []
cpso_dissipative_generalization_error = []
pso_errors = []
pso_error = []
pso_generalization_error = []

iterations = 5000
samples = 30

NUM_PARTICLES_Y = 5
NUM_PARTICLES_X = 5
INERTIA_WEIGHT = 0.729844
COGNITIVE_CONSTANT = 1.496180
SOCIAL_CONSTANT = 1.496180
BOUNDS = Bounds(-10, 10)
INITIAL_POSITION = Bounds(-5, 5)

DESC = 'Glass'
DATA_SET_FUNC = dataSetTool.getGlassDataSets
DATA_SET_FILE_LOC = '../dataSets/glass/glass.data'
HIDDEN_LAYER_NEURONS = [12]

# DESC = 'Iris'
# DATA_SET_FUNC = dataSetTool.getIrisDataSets
# DATA_SET_FILE_LOC = '../dataSets/iris/iris.data'
# HIDDEN_LAYER_NEURONS = [8]

# DESC = 'Wine'
# DATA_SET_FUNC = dataSetTool.getWineDataSets
# DATA_SET_FILE_LOC = '../dataSets/wine/wine.data'
# HIDDEN_LAYER_NEURONS = [10]

# DESC = 'Diabetes'
# DATA_SET_FUNC = dataSetTool.getPrimaIndiansDiabetesSets
# DATA_SET_FILE_LOC = '../dataSets/pima-indians-diabetes/pima-indians-diabetes.data'
# HIDDEN_LAYER_NEURONS = [20]

# DESC = 'Heart'
# DATA_SET_FUNC = dataSetTool.getHeartDataSets
# DATA_SET_FILE_LOC = '../dataSets/heart/processed.cleveland.data'
# HIDDEN_LAYER_NEURONS = [10]


# None, 0.1
V_MAX = None

dataSetArray = []
for _ in range(iterations):
    dataSetArray.append(DATA_SET_FUNC(DATA_SET_FILE_LOC))

print('Vmax: ', V_MAX)
print('Data Set: ', DESC)
print('Tinkerbell:')
for i in range(samples):
    psonn = VNPSONN()
    psonn.training, psonn.testing, psonn.generalization = dataSetArray[i]
    psonn.bounds = BOUNDS
    psonn.initialPosition = INITIAL_POSITION
    psonn.createNeuralNetwork(HIDDEN_LAYER_NEURONS)
    psonn.num_particles_x = NUM_PARTICLES_X
    psonn.num_particles_y = NUM_PARTICLES_Y
    psonn.inertia_weight = INERTIA_WEIGHT
    psonn.cognitiveConstant = COGNITIVE_CONSTANT
    psonn.socialConstant = SOCIAL_CONSTANT
    psonn.vmax = V_MAX

    from mlpy.numberGenerator.chaos.tinkerbell import Tinkerbell
    psonn.numberGenerator = Tinkerbell()

    psonn.color = 'red'
    trainingErrors, trainingError, generalizationError = psonn.train(iterations)

    cpso_tinkerbell_errors.append(trainingErrors)
    cpso_tinkerbell_error.append(trainingError)
    cpso_tinkerbell_generalization_error.append(generalizationError)

print('Lozi:')
for i in range(samples):
    psonn = VNPSONN()
    psonn.training, psonn.testing, psonn.generalization = dataSetArray[i]
    psonn.bounds = BOUNDS
    psonn.initialPosition = INITIAL_POSITION
    psonn.createNeuralNetwork(HIDDEN_LAYER_NEURONS)
    psonn.num_particles_x = NUM_PARTICLES_X
    psonn.num_particles_y = NUM_PARTICLES_Y
    psonn.inertia_weight = INERTIA_WEIGHT
    psonn.cognitiveConstant = COGNITIVE_CONSTANT
    psonn.socialConstant = SOCIAL_CONSTANT
    psonn.vmax = V_MAX

    from mlpy.numberGenerator.chaos.lozi import Lozi
    psonn.numberGenerator = Lozi()

    psonn.color = 'green'
    trainingErrors, trainingError, generalizationError = psonn.train(iterations)

    cpso_lozi_errors.append(trainingErrors)
    cpso_lozi_error.append(trainingError)
    cpso_lozi_generalization_error.append(generalizationError)

print('Dissipative:')
for i in range(samples):
    psonn = VNPSONN()
    psonn.training, psonn.testing, psonn.generalization = dataSetArray[i]
    psonn.bounds = BOUNDS
    psonn.initialPosition = INITIAL_POSITION
    psonn.createNeuralNetwork(HIDDEN_LAYER_NEURONS)
    psonn.num_particles_x = NUM_PARTICLES_X
    psonn.num_particles_y = NUM_PARTICLES_Y
    psonn.inertia_weight = INERTIA_WEIGHT
    psonn.cognitiveConstant = COGNITIVE_CONSTANT
    psonn.socialConstant = SOCIAL_CONSTANT
    psonn.vmax = V_MAX

    from mlpy.numberGenerator.chaos.dissipative import Dissipative
    psonn.numberGenerator = Dissipative()

    psonn.color = 'blue'
    trainingErrors, trainingError, generalizationError = psonn.train(iterations)

    cpso_dissipative_errors.append(trainingErrors)
    cpso_dissipative_error.append(trainingError)
    cpso_dissipative_generalization_error.append(generalizationError)

print('Random:')
for i in range(samples):
    psonn = VNPSONN()
    psonn.training, psonn.testing, psonn.generalization = dataSetArray[i]
    psonn.bounds = BOUNDS
    psonn.initialPosition = INITIAL_POSITION
    psonn.createNeuralNetwork(HIDDEN_LAYER_NEURONS)
    psonn.num_particles_x = NUM_PARTICLES_X
    psonn.num_particles_y = NUM_PARTICLES_Y
    psonn.inertia_weight = INERTIA_WEIGHT
    psonn.cognitiveConstant = COGNITIVE_CONSTANT
    psonn.socialConstant = SOCIAL_CONSTANT
    psonn.vmax = V_MAX

    psonn.color = 'black'
    trainingErrors, trainingError, generalizationError = psonn.train(iterations)

    pso_errors.append(trainingErrors)
    pso_error.append(trainingError)
    pso_generalization_error.append(generalizationError)

iterations = [y[1] for y in pso_errors[0]]

# Tinkerbell
cpso_tinkerbell_errors_no_iteration = [[y[0] for y in x] for x in cpso_tinkerbell_errors]
cpso_tinkerbell_errors_mean = np.mean(cpso_tinkerbell_errors_no_iteration, axis=0)

cpso_tinkerbell_error_mean = np.mean(cpso_tinkerbell_error)
cpso_tinkerbell_error_std = np.std(cpso_tinkerbell_error)

cpso_tinkerbell_generalization_error_mean = np.mean(cpso_tinkerbell_generalization_error)
cpso_tinkerbell_generalization_error_std = np.std(cpso_tinkerbell_generalization_error)

cpso_tinkerbell_generalization_factor = np.divide(cpso_tinkerbell_generalization_error, cpso_tinkerbell_error)
cpso_tinkerbell_generalization_factor_mean = np.mean(cpso_tinkerbell_generalization_factor)
cpso_tinkerbell_generalization_factor_std = np.std(cpso_tinkerbell_generalization_factor)

# Lozi
cpso_lozi_errors_no_iteration = [[y[0] for y in x] for x in cpso_lozi_errors]
cpso_lozi_errors_mean = np.mean(cpso_lozi_errors_no_iteration, axis=0)

cpso_lozi_error_mean = np.mean(cpso_lozi_error)
cpso_lozi_error_std = np.std(cpso_lozi_error)

cpso_lozi_generalization_error_mean = np.mean(cpso_lozi_generalization_error)
cpso_lozi_generalization_error_std = np.std(cpso_lozi_generalization_error)

cpso_lozi_generalization_factor = np.divide(cpso_lozi_generalization_error, cpso_lozi_error)
cpso_lozi_generalization_factor_mean = np.mean(cpso_lozi_generalization_factor)
cpso_lozi_generalization_factor_std = np.std(cpso_lozi_generalization_factor)

# Dissipative
cpso_dissipative_errors_no_iteration = [[y[0] for y in x] for x in cpso_dissipative_errors]
cpso_dissipative_errors_mean = np.mean(cpso_dissipative_errors_no_iteration, axis=0)

cpso_dissipative_error_mean = np.mean(cpso_dissipative_error)
cpso_dissipative_error_std = np.std(cpso_dissipative_error)

cpso_dissipative_generalization_error_mean = np.mean(cpso_dissipative_generalization_error)
cpso_dissipative_generalization_error_std = np.std(cpso_dissipative_generalization_error)

cpso_dissipative_generalization_factor = np.divide(cpso_dissipative_generalization_error, cpso_dissipative_error)
cpso_dissipative_generalization_factor_mean = np.mean(cpso_dissipative_generalization_factor)
cpso_dissipative_generalization_factor_std = np.std(cpso_dissipative_generalization_factor)

# Random
pso_errors_no_iteration = [[y[0] for y in x] for x in pso_errors]
pso_errors_mean = np.mean(pso_errors_no_iteration, axis=0)

pso_error_mean = np.mean(pso_error)
pso_error_std = np.std(pso_error)

pso_generalization_error_mean = np.mean(pso_generalization_error)
pso_generalization_error_std = np.std(pso_generalization_error)

pso_generalization_factor = np.divide(pso_generalization_error, pso_error)
pso_generalization_factor_mean = np.mean(pso_generalization_factor)
pso_generalization_factor_std = np.std(pso_generalization_factor)

print('- Tinkerbell:')
print(cpso_tinkerbell_error_mean)
print('(' + str(cpso_tinkerbell_error_std) + ')')
print(cpso_tinkerbell_generalization_error_mean)
print('(' + str(cpso_tinkerbell_generalization_error_std) + ')')
print(cpso_tinkerbell_generalization_factor_mean)
print('(' + str(cpso_tinkerbell_generalization_factor_std) + ')')

print('- Lozi:')
print(cpso_lozi_error_mean)
print('(' + str(cpso_lozi_error_std) + ')')
print(cpso_lozi_generalization_error_mean)
print('(' + str(cpso_lozi_generalization_error_std) + ')')
print(cpso_lozi_generalization_factor_mean)
print('(' + str(cpso_lozi_generalization_factor_std) + ')')

print('- Dissipative:')
print(cpso_dissipative_error_mean)
print('(' + str(cpso_dissipative_error_std) + ')')
print(cpso_dissipative_generalization_error_mean)
print('(' + str(cpso_dissipative_generalization_error_std) + ')')
print(cpso_dissipative_generalization_factor_mean)
print('(' + str(cpso_dissipative_generalization_factor_std) + ')')

print('- Random:')
print(pso_error_mean)
print('(' + str(pso_error_std) + ')')
print(pso_generalization_error_mean)
print('(' + str(pso_generalization_error_std) + ')')
print(pso_generalization_factor_mean)
print('(' + str(pso_generalization_factor_std) + ')')

plt.close()

fig = plt.figure()
plt.grid(1)
plt.ylim([0, 0.35])
plt.xlim([0, 5000])
plt.ion()
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
fig.suptitle('Result' + DESC + 'Vmax' + str(V_MAX))
tinkerbell, = plt.plot(iterations, cpso_tinkerbell_errors_mean, 'r--', linewidth=1, markersize=3)
lozi, = plt.plot(iterations, cpso_lozi_errors_mean, 'b:', linewidth=1, markersize=3)
dissipative, = plt.plot(iterations, cpso_dissipative_errors_mean, 'g+', linewidth=1, markersize=3)
random, = plt.plot(iterations, pso_errors_mean, 'k-', linewidth=1, markersize=3)
plt.legend([tinkerbell, lozi, dissipative, random], ['Tinkerbell', 'Lozi', 'Dissipative', 'Uniform'])
fig.savefig('Result' + DESC + 'Vmax' + str(V_MAX) + '.png')
plt.show(5)