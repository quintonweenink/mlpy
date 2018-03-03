import matplotlib.pyplot as plt
import numpy as np

from mlpy.particleSwarmOptimization.structure.particle import Particle

np.set_printoptions(suppress=True)

class PSO(object):

    def __init__(self):
        self.num_dimensions = None

        self.group_best_error = float('inf')
        self.group_best_position = None

        self.best_error = float('inf')
        self.best_position = None

        self.num_particles = None
        self.bounds = None
        self.weight = None
        self.cognitiveConstant = None
        self.socialConstant = None

        self.vmax = None

        self.swarm = []
        self.initialPosition = None

        self.color = None

        self.error = None

    def getGlobalBest(self):
        self.best_error = float('inf')
        for particle in self.swarm:
            if particle.error < self.group_best_error:
                self.group_best_position = np.array(particle.position)
                self.group_best_error = particle.error
            # Get current best as well
            if particle.error < self.best_error:
                self.best_position = np.array(particle.position)
                self.best_error = particle.error

        return self.group_best_position

    def createParticles(self):
        for i in range(self.num_particles):
            self.swarm.append(Particle(self.bounds, self.weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(
                self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.swarm[i].initPos(position, velocity)

    def loopOverParticles(self):
        for j in range(self.num_particles):
            self.swarm[j].error = self.error(self.swarm[j].position)

            self.swarm[j].getPersonalBest()

        self.getGlobalBest()

        for j in range(self.num_particles):
            self.swarm[j].update_velocity(self.group_best_position)
            self.swarm[j].update_position()


    def train(self, iterations):
        self.createParticles()

        trainingErrors = []

        plt.grid(1)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        # plt.ylim([0, 1])
        plt.ion()

        for x in range(iterations):

            self.loopOverParticles()

            if (x % 100 == 0):
                trainingErrors.append([self.best_error, x])
                plt.scatter(x, self.best_error, color=self.color, s=4, label="test1")
                plt.pause(0.0001)
                plt.show()

        print("Best error:\t\t\t" + str(self.group_best_error))
        print("Current best error:\t" + str(self.best_error) + "\n")

        return trainingErrors, self.group_best_error
