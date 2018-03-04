import numpy as np

from mlpy.particleSwarmOptimization.structure.particle import Particle

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


    def train(self, iterations, sampleSize):
        self.createParticles()

        trainingErrors = []

        for x in range(iterations):

            self.loopOverParticles()

            if (x % sampleSize == 0):
                trainingErrors.append([self.group_best_position, x])

        trainingErrors.append([self.group_best_error, iterations])

        return trainingErrors, self.group_best_error
