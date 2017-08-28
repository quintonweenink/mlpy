import random

import numpy as np
import matplotlib.pyplot as plt

from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.layer import Layer

import math

from chaos.lozi import Lozi
from src.particleSwarmOptimization.cpso import CPSO
from src.particleSwarmOptimization.pso import PSO
from src.particleSwarmOptimization.structure.particle import Particle
from src.particleSwarmOptimization.structure.bounds import Bounds


bounds = Bounds(-10, 10)

plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.ylim([0,1])
plt.legend(['error'])
plt.ion()

l_rate = 0.5

inputLayer = Layer(size = 4, prev = None, l_rate = l_rate, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 6, prev = inputLayer, l_rate = l_rate, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 3, prev = hiddenLayer, l_rate = l_rate, bias = False, label = "Output layer")

fnn = NeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

from numpy import genfromtxt
input = np.array(genfromtxt('neuralNetwork/dataSet/iris.data', delimiter=',', usecols=(0, 1, 2, 3)))
outputClassification = np.array(genfromtxt('neuralNetwork/dataSet/iris.data', dtype=str, delimiter=',', usecols=4))

classifications = []

for item in outputClassification:
    notInList = True
    for classification in classifications:
        if(classification == item):
            notInList = False
    if notInList:
        classifications.append(item)


output = []

for item in outputClassification:
    target = []
    for classification in classifications:
        if item == classification:
            target.append(1)
        else:
            target.append(0)
    output.append(target)

output = np.array(output)

input_target = []

for i in range(len(input)):
    input_target.append((input[i], output[i]))

random.shuffle(input_target)

training = input_target[:int(len(input_target)/2)]
testing = input_target[int(len(input_target)/2):]

errors = []

num_particles = 15
maxiter = 30
weight = 0.5
cognitiveConstant = 1
socialConstant = 2

num_dimensions = len(fnn.getAllWeights())

numberGenerator = Lozi()

i = 0
while i < 1000:
    print(numberGenerator.random())
    i += 1


pso = PSO(None, num_dimensions, bounds, numberGenerator,
                   num_particles, maxiter, weight, cognitiveConstant, socialConstant)

for i in range(pso.num_particles):
    pso.swarm.append(
        Particle(bounds, numberGenerator, num_dimensions, None, weight, cognitiveConstant, socialConstant))
    pso.swarm[i].initPos()

for i in range(8000):
    mod = i % len(training)
    in_out = training[mod]
    for j in range(pso.num_particles):
        pso.swarm[j].err_i = fnn.fire(np.array([in_out[0]]))

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(self.pos_best_g)
        pso.swarm[j].update_position()
    print(pso)

print('FINAL:')
print(pso)

for i in range(8000):
    mod = i % len(training)
    in_out = training[mod]
    result = fnn.fire(np.array([in_out[0]]))
    i_error = fnn.backPropagation(np.array([in_out[1]]))

    if (i % 50) == 0:
        #print("Error:" + str(fnn))
        errors.append(abs(i_error[0][0]))
        plt.scatter(i, abs(i_error[0][0]))
        plt.pause(0.001)
        plt.draw()


plt.pause(5)
print(fnn)
