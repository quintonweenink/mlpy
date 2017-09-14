import numpy as np
import matplotlib.pyplot as plt

from dataSetTool import DataSetTool
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from numberGenerator.bounds import Bounds
from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.structure.particle import Particle

np.set_printoptions(suppress=True)

# Get data set
dataSetTool = DataSetTool()
training, testing = dataSetTool.getIrisDataSets('../dataSet/iris/iris.data')

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,0.5])
plt.ion()

errors = []
bounds = Bounds(-10, 10)

# Create neural network
l_rate = None
inputLayer = Layer(bounds, size = 4, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(bounds, size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(bounds, size = 3, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")
fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

# Create the pso with the nn weights
num_particles = 30
inertia_weight = 0.729
cognitiveConstant = 1.49445
socialConstant = 1.49445
num_dimensions = len(fnn.getAllWeights())
# Configure PSO
pso = PSO(bounds, num_particles, inertia_weight, cognitiveConstant, socialConstant)


group_training = np.array([input[0] for input in training])
group_target = np.array([output[1] for output in training])

# Create particles
for i in range(pso.num_particles):
    pso.swarm.append(Particle(bounds, inertia_weight, cognitiveConstant, socialConstant))
    pso.swarm[i].initPos((bounds.maxBound - bounds.minBound) * np.random.random(num_dimensions) + bounds.minBound)

# Iterate over training data
for i in range(400):
    # Get the iteration data
    mod = i % len(training)
    in_out = training[mod]
    # Loop over particles
    for j in range(pso.num_particles):
        # Set weights according to pso particle
        fnn.setAllWeights(pso.swarm[j].position)

        # Fire the neural network and calculate error
        result = fnn.fire(np.array(group_training))
        difference = group_target - result
        error = np.mean(np.square(difference))
        errors.append(error)

        pso.swarm[j].error = error

        # Get & set personal best
        pso.swarm[j].getPersonalBest()

        # Print results
        print(j, np.array(pso.swarm[j].error))

    # Get & set global best
    pso.getGlobalBest()

    plt.scatter(i, pso.best_error, color='blue', s=4, label="test1")
    plt.pause(0.0001)
    plt.show()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

    if(i % 1 == 0):
        print("Best error:\t\t\t" + str(pso.group_best_error))
        print("Current best error:\t" + str(pso.best_error) + "\n")

fnn.setAllWeights(pso.group_best_position)
result = fnn.fire(np.array(group_training))
error = group_target - result
print("RESULT: " + str(result))
print("TARGET: " + str(group_target))
print("ERROR: " + str(error))

