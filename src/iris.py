import random

import numpy as np
import matplotlib.pyplot as plt

from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.layer import Layer

import math

from src.particleSwarmOptimization.numberGenerator.chaos.lozi import Lozi
from src.particleSwarmOptimization.numberGenerator.rng import RNG
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

num_particles = 5
maxiter = 30
weight = 1
cognitiveConstant = 1
socialConstant = 0.1

num_dimensions = len(fnn.getAllWeights())

numberGenerator = RNG()

pso = PSO(None, num_dimensions, bounds, numberGenerator,
                   num_particles, maxiter, weight, cognitiveConstant, socialConstant)

for i in range(pso.num_particles):
    pso.swarm.append(
        Particle(bounds, numberGenerator, num_dimensions, None, weight, cognitiveConstant, socialConstant))
    pso.swarm[i].initPos()

setosa = np.array([[5.0,3.3,1.4,0.2]])#Iris-setosa
setosa_o = np.array([[1,0,0]])
versicolor = np.array([[5.7,2.8,4.1,1.3]])#Iris-versicolor
versicolor_o = np.array([[0,1,0]])
virginica = np.array([[5.9,3.0,5.1,1.8]])#Iris-virginica
virginica_o = np.array([[0,0,1]])

print("FIRE: " + str(fnn.fire(setosa)))
print("OUT: " + str(setosa_o))
print("FIRE: " + str(fnn.fire(versicolor)))
print("OUT: " + str(versicolor_o))
print("FIRE: " + str(fnn.fire(virginica)))
print("OUT: " + str(virginica_o))


for i in range(20):
    mod = i % len(training)
    in_out = training[mod]
    print(in_out[1])
    for j in range(pso.num_particles):
        fnn.setAllWeights(pso.swarm[j].position_i)
        result = fnn.fire(np.array([in_out[0]]))[0]
        error = in_out[1] - result
        pso.swarm[j].err_i = np.sum(abs(error))
        print(j, pso.swarm[j].getPersonalBest())
        print("Result: \t" + str(result))
        print("Error: \t\t" + str(error))
        print("Abs Error: \t" + str(abs(error)))
        print("Final error: \t" + str(pso.swarm[j].err_i))
        # Need to talk to Anna about PSO going to satisfy one classification in stead of solving multiple classifications. I think this would be solved by using multiple classifications at once training sets at once but that doesn't seem like a viable solution.

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.pos_best_g)
        pso.swarm[j].update_position()

    print("Best error: " + str(pso.err_best_g) + "\n")


print('FINAL:')
print(pso)

print(fnn)

print("FIRE: " + str(fnn.fire(setosa)))
print("OUT: " + str(setosa_o))
print("FIRE: " + str(fnn.fire(versicolor)))
print("OUT: " + str(versicolor_o))
print("FIRE: " + str(fnn.fire(virginica)))
print("OUT: " + str(virginica_o))
