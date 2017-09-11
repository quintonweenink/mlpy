import numpy as np

from numberGenerator.rng import RNG
from numberGenerator.bounds import Bounds
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer
from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.structure.particle import Particle


np.set_printoptions(suppress=True)

input = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])

target = np.array([[0],
                   [1],
                   [1],
                   [0]])

training = []
for x, y in zip(input, target):
    training.append((x, y))

errors = []
bounds = Bounds(-5, 5)

# Create neural network
l_rate = 0.5
inputLayer = Layer(bounds, size = 2, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(bounds, size = 4, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(bounds, size = 1, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")
fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

# Create the pso with the nn weights
num_particles = 20
maxiter = None
weight = 0.7
cognitiveConstant = 1.4
socialConstant = 1.4
num_dimensions = len(fnn.getAllWeights())
numberGenerator = RNG()
pso = PSO(bounds, numberGenerator, num_particles, weight, cognitiveConstant, socialConstant)
for i in range(pso.num_particles):
    pso.swarm.append(Particle(bounds, numberGenerator, weight, cognitiveConstant, socialConstant))
    pso.swarm[i].initPos((bounds.maxBound - bounds.minBound) * np.random.random(num_dimensions) + bounds.minBound)


# Iterate over training data
for i in range(400):
    for j in range(pso.num_particles):
        fnn.setAllWeights(pso.swarm[j].position)
        result = fnn.fire(np.array(input))
        error = target - result
        pso.swarm[j].error = np.mean(np.square(error))
        pso.swarm[j].getPersonalBest()
        print(j, np.array(pso.swarm[j].error))

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

    print("Best error: " + str(pso.group_best_error))
    print("Current best error: " + str(pso.group_best_error) + "\n")


print('FINAL:')
print(pso)
print(fnn)

print("FIRE: " + str(fnn.fire(np.array([input[0]]))))
print("FIRE: " + str(fnn.fire(np.array([input[1]]))))
print("FIRE: " + str(fnn.fire(np.array([input[2]]))))
print("FIRE: " + str(fnn.fire(np.array([input[3]]))))
