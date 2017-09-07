import random

import numpy as np
import matplotlib.pyplot as plt
import sys

from src.particleSwarmOptimization.numberGenerator.chaos.lozi import Lozi
from src.particleSwarmOptimization.numberGenerator.rng import RNG
from src.particleSwarmOptimization.cpso import CPSO
from src.particleSwarmOptimization.pso import PSO
from src.particleSwarmOptimization.structure.particle import Particle
from src.particleSwarmOptimization.structure.bounds import Bounds
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.layer import Layer
from neuralNetwork.dataSet.dataSetTool import DataSetTool

np.set_printoptions(suppress=True)

# Get data set
dataSetTool = DataSetTool()
training, testing = dataSetTool.getIrisDataSets('neuralNetwork/dataSet/iris.data')

errors = []
bounds = Bounds(-10, 10)

# Create neural network
l_rate = None
inputLayer = Layer(size = 4, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 3, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")
fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

# Create the pso with the nn weights
num_particles = 30
inertia_weight = 0.06
cognitiveConstant = 0.9
socialConstant = 0.9
num_dimensions = len(fnn.getAllWeights())
numberGenerator = RNG()
# Configure PSO
pso = PSO(bounds, numberGenerator, num_particles, inertia_weight, cognitiveConstant, socialConstant)

result = fnn.fire(np.array([[0.2, 0.2, 0.2, 0.2]]))[0]
print(result)
fnn.setAllWeights(20 * np.random.random(num_dimensions) - 10)
result = fnn.fire(np.array([[0.2, 0.2, 0.2, 0.2]]))[0]
print(result)
weights = fnn.getAllWeights()
fnn.setAllWeights(weights)
result = fnn.fire(np.array([[0.2, 0.2, 0.2, 0.2]]))[0]
print(result)


# sys.exit(0)
# Create particles
for i in range(pso.num_particles):
    pso.swarm.append(Particle(bounds, numberGenerator, inertia_weight, cognitiveConstant, socialConstant))
    pso.swarm[i].initPos(5 * np.random.random(num_dimensions) - 2.5)

# Iterate over training data
for i in range(20):
    # Get the iteration data
    mod = i % len(training)
    in_out = training[mod]
    # Loop over particles
    for j in range(pso.num_particles):
        # Set weights according to pso particle
        fnn.setAllWeights(pso.swarm[j].position_i)

        # Fire the neural network and calculate error
        result = fnn.fire(np.array([in_out[0]]))[0]
        error = in_out[1] - result
        pso.swarm[j].err_i = np.mean(np.square(error)) * 100

        # Get & set personal best
        pso.swarm[j].getPersonalBest()
        print(j, np.array(pso.swarm[j].err_i))
        print("Result: \t" + str(np.array(result)))
        print("Error: \t\t" + str(np.array(error)))

    # Get & set global best
    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.pos_best_g)
        pso.swarm[j].update_position()

    print(in_out[1])
    print("Best error:\t\t\t" + str(pso.err_best_g))
    print("Current best error:\t" + str(pso.err_best_i) + "\n")

# Iterate over training data
# for i in range(len(testing)):
#     in_out = training[i]
#     print(in_out[1])
#     # Loop over particles
#     for j in range(pso.num_particles):
#         # Set weights according to pso particle
#         fnn.setAllWeights(pso.swarm[j].position_i)
#
#         # Fire the neural network and calculate error
#         result = fnn.fire(np.array([in_out[0]]))[0]
#         error = in_out[1] - result
#
#         # Get & set personal best
#         print(j, np.array(pso.swarm[j].err_i))
#         print("Result: \t" + str(np.array(result)))
#         print("Error: \t\t" + str(np.array(error)))


print('FINAL:')
print(pso)
print(fnn)

# PSO NN Error calculation
# Follow Anna paper
# Initial velocity
#