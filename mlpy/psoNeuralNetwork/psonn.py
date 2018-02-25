import matplotlib.pyplot as plt
import numpy as np

from mlpy.particleSwarmOptimization.pso import PSO
from mlpy.neuralNetwork.feedForwardNeuralNetwork import NeuralNetwork
from mlpy.neuralNetwork.structure.layer import Layer
from mlpy.particleSwarmOptimization.structure.particle import Particle

np.set_printoptions(suppress=True)

class PSONN(PSO):
    def __init__(self):
        super(PSONN, self).__init__()

        self.nn = None

        self.batch_training_input = None
        self.batch_training_target = None

        self.batch_generalization_input = None
        self.batch_generalization_target = None

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

    def createParticles(self):
        for i in range(self.num_particles):
            self.swarm.append(
                Particle(self.bounds, self.weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(
                self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.swarm[i].initPos(position, velocity)

    def loopOverParticles(self):
        # Loop over particles
        for i, particle in enumerate(self.swarm):
            # Set weights according to mlpy particle
            self.nn.setAllWeights(self.swarm[i].position)

            result = self.nn.fire(np.array(self.batch_training_input))
            difference = self.batch_training_target - result
            error = np.mean(np.square(difference))

            self.swarm[i].error = error

            # Get & set personal best
            self.swarm[i].getPersonalBest()

        # Get & set global best
        self.getGlobalBest()

        for j in range(self.num_particles):
            self.swarm[j].update_velocity(self.group_best_position)
            self.swarm[j].update_position(self.vmax)


    def train(self, iterations):
        self.batch_training_input = np.array([input[0] for input in self.training])
        self.batch_training_target = np.array([output[1] for output in self.training])

        self.batch_generalization_input = np.array([input[0] for input in self.generalization])
        self.batch_generalization_target = np.array([output[1] for output in self.generalization])

        self.num_dimensions = len(self.nn.getAllWeights())
        #print('Dimensions', self.num_dimensions)

        self.createParticles()

        trainingErrors = []

        plt.grid(1)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.ylim([0, 1])
        plt.ion()

        # Iterate over training data
        for x in range(iterations):

            self.loopOverParticles()

            if (x % 100 == 0):
                trainingErrors.append([self.best_error, x])
                plt.scatter(x, self.best_error, color=self.color, s=4, label="test1")
                plt.pause(0.0001)
                plt.show()

        correct = 0

        self.nn.setAllWeights(self.best_position)

        result = self.nn.fire(np.array(self.batch_training_input))
        difference = self.batch_training_target - result
        trainingError = np.mean(np.square(difference))
        trainingErrors.append([trainingError, 5000])

        result = self.nn.fire(np.array(self.batch_generalization_input))
        difference = self.batch_generalization_target - result
        generalizationError = np.mean(np.square(difference))

        for i in range(len(self.generalization)):
            in_out = self.generalization[i]
            result = self.nn.fire(np.array([in_out[0]]))

            if np.argmax(result) == np.argmax(in_out[1]):
                correct += 1

        print(str(correct / len(self.generalization)))

        return trainingErrors, trainingError, generalizationError

