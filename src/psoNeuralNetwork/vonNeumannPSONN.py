import numpy as np
import matplotlib.pyplot as plt

from numberGenerator.chaos.cprng import CPRNG
from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.structure.particle import Particle
from particleSwarmOptimization.structure.chaoticParticle import ChaoticParticle
from neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from neuralNetwork.structure.layer import Layer

np.set_printoptions(suppress=True)

class VNPSONN(object):
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

        self.vmax = None

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
                row = []
                for j in range(self.pso.num_particles):
                    particle = ChaoticParticle(self.bounds, self.numberGenerator, self.inertia_weight, self.cognitiveConstant, self.socialConstant)
                    particle.initPos((self.bounds.maxBound - self.bounds.minBound) * np.random.random(self.num_dimensions) + self.bounds.minBound)
                    row.append(particle)

                self.pso.swarm.append(row)
        else:
            for i in range(self.pso.num_particles):
                row = []
                for j in range(self.pso.num_particles):
                    particle = Particle(self.bounds, self.inertia_weight, self.cognitiveConstant, self.socialConstant)
                    particle.initPos((self.bounds.maxBound - self.bounds.minBound) * np.random.random(self.num_dimensions) + self.bounds.minBound)
                    row.append(particle)

                self.pso.swarm.append(row)

        for i in range(self.pso.num_particles):
            for j in range(self.pso.num_particles):
                if i > 0:  # We can go west
                    self.pso.swarm[i][j].neighbourhood.append(self.pso.swarm[i - 1][j])
                if i < self.pso.num_particles - 1:  # We can go east
                    self.pso.swarm[i][j].neighbourhood.append(self.pso.swarm[i + 1][j])
                if j > 0:  # We can go north
                    self.pso.swarm[i][j].neighbourhood.append(self.pso.swarm[i][j - 1])
                if j < self.pso.num_particles - 1:  # We can go south
                    self.pso.swarm[i][j].neighbourhood.append(self.pso.swarm[i][j + 1])

        self.batch_training_input = np.array([input[0] for input in self.training])
        self.batch_training_target = np.array([output[1] for output in self.training])

        errors = []

        plt.grid(1)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.ylim([0, 1])
        plt.ion()

        # Iterate over training data
        for x in range(8000):

            # Loop over particles
            for i, row in enumerate(self.pso.swarm):
                for j, col in enumerate(row):
                    # Set weights according to pso particle
                    self.nn.setAllWeights(self.pso.swarm[i][j].position)

                    result = self.nn.fire(np.array(self.batch_training_input))
                    difference = self.batch_training_target - result
                    error = np.mean(np.square(difference))
                    errors.append(error)

                    self.pso.swarm[i][j].error = error

                    # Get & set personal best
                    self.pso.swarm[i][j].getPersonalBest()

                    # Print results
                    # print(j, np.array(self.pso.swarm[j].error))

            for i in range(self.pso.num_particles):
                for j in range(self.pso.num_particles):
                    particle = self.pso.swarm[i][j]
                    neighbourhoodBest = particle.error
                    neighbourhoodBestPos = particle.position

                    for neighbour in particle.neighbourhood:
                        if abs(neighbour.error) < abs(neighbourhoodBest):
                            neighbourhoodBestPos = np.array(neighbour.position)
                            neighbourhoodBest = neighbour.error
                        # Get current global best as well
                        if abs(neighbour.error) < abs(particle.best_error):
                            self.pso.best_position = np.array(particle.position)
                            self.pso.best_error = particle.error

                    self.pso.swarm[i][j].update_velocity(neighbourhoodBestPos)
                    self.pso.swarm[i][j].update_position(self.vmax)

            if (x % 53 == 0):
                plt.scatter(x, self.pso.best_error, color='blue', s=4, label="test1")
                plt.pause(0.0001)
                plt.show()

                print("Current best error:\t" + str(self.pso.best_error) + "\n")

        correct = 0

        self.nn.setAllWeights(self.pso.best_position)

        for i in range(len(self.testing)):
            in_out = self.testing[i]
            result = self.nn.fire(np.array([in_out[0]]))

            print(result)
            print(in_out[1])
            print()

            if np.argmax(result) == np.argmax(in_out[1]):
                correct += 1

        print("Classification accuracy: ", str(100 * (correct / len(self.testing))  ) + "%")

        return errors

