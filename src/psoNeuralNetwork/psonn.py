import numpy as np
import matplotlib.pyplot as plt

from numberGenerator.chaos.cprng import CPRNG
from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.structure.particle import Particle
from particleSwarmOptimization.structure.chaoticParticle import ChaoticParticle
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer

np.set_printoptions(suppress=True)

class PSONN(object):
    def __init__(self):
        self.nn = None
        self.pso = None

        self.bounds = None
        self.training = None
        self.testing = None
        self.num_particles = None
        self.inertia_weight = None
        self.cognitiveConstant = None
        self.socialConstant = None
        self.numberGenerator = None

        self.batch_training_input = None
        self.batch_training_target = None

    def createNeuralNetwork(self, hiddenArr):
        l_rate = None
        self.nn = NeuralNetwork()
        self.nn.appendLayer(Layer(self.bounds, size=len(self.training[0][0]), prev=None, l_rate=l_rate, bias=True,
                           label="Input layer"))
        for i in hiddenArr:
            prevLayer = self.nn.layers[len(self.nn.layers) - 1]
            self.nn.appendLayer(Layer(self.bounds, size=i, prev=prevLayer, l_rate=l_rate, bias=True, label="Hidden layer"))

        prevLayer = self.nn.layers[len(self.nn.layers) - 1]
        self.nn.appendLayer(Layer(self.bounds, size=len(self.training[0][1]), prev=prevLayer, l_rate=l_rate, bias=False, label="Output layer"))

    def train(self):
        self.pso = PSO(self.bounds, self.num_particles, self.inertia_weight, self.cognitiveConstant, self.socialConstant)

        self.batch_training_input = np.array([input[0] for input in self.training])
        self.batch_training_target = np.array([output[1] for output in self.training])

        self.num_dimensions = len(self.nn.getAllWeights())

        # Create particles
        if isinstance(self.numberGenerator, CPRNG):
            for i in range(self.pso.num_particles):
                self.pso.swarm.append(ChaoticParticle(self.bounds, self.numberGenerator, self.inertia_weight, self.cognitiveConstant, self.socialConstant))
                self.pso.swarm[i].initPos((self.bounds.maxBound - self.bounds.minBound) * np.random.random(self.num_dimensions) + self.bounds.minBound)

        else:
            for i in range(self.pso.num_particles):
                self.pso.swarm.append(Particle(self.bounds, self.inertia_weight, self.cognitiveConstant, self.socialConstant))
                self.pso.swarm[i].initPos((self.bounds.maxBound - self.bounds.minBound) * np.random.random(self.num_dimensions) + self.bounds.minBound)

        self.batch_training_input = np.array([input[0] for input in self.training])
        self.batch_training_target = np.array([output[1] for output in self.training])

        errors = []

        # Iterate over training data
        for i in range(400):
            # Get the iteration data
            mod = i % len(self.training)
            in_out = self.training[mod]
            # Loop over particles
            for j in range(self.pso.num_particles):
                # Set weights according to pso particle
                self.nn.setAllWeights(self.pso.swarm[j].position)

                result = self.nn.fire(np.array(self.batch_training_input))
                difference = self.batch_training_target - result
                error = np.mean(np.square(difference))
                errors.append(error)

                self.pso.swarm[j].error = error

                # Get & set personal best
                self.pso.swarm[j].getPersonalBest()

                # Print results
                print(j, np.array(self.pso.swarm[j].error))

            # Get & set global best
            self.pso.getGlobalBest()

            # plt.scatter(i, pso.best_error, color='blue', s=4, label="test1")
            # plt.pause(0.0001)
            # plt.show()

            for j in range(self.pso.num_particles):
                self.pso.swarm[j].update_velocity(self.pso.group_best_position)
                self.pso.swarm[j].update_position()

            if (i % 1 == 0):
                print("Best error:\t\t\t" + str(self.pso.group_best_error))
                print("Current best error:\t" + str(self.pso.best_error) + "\n")

        return errors

