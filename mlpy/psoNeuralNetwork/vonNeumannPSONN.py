import numpy as np

from mlpy.particleSwarmOptimization.structure.particle import Particle

from mlpy.psoNeuralNetwork.psonn import PSONN

class VNPSONN(PSONN):
    def __init__(self):
        super(VNPSONN, self).__init__()

        self.num_particles_x = None
        self.num_particles_y = None

    def createParticles(self):
        for i in range(self.num_particles_x):
            row = []
            for j in range(self.num_particles_y):
                particle = Particle(self.bounds, self.inertia_weight, self.cognitiveConstant, self.socialConstant)
                position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(self.num_dimensions) + self.initialPosition.minBound
                velocity = np.zeros(self.num_dimensions)
                particle.initPos(position, velocity)
                row.append(particle)

            self.swarm.append(row)

        self.linkParticles()

    def linkParticles(self):
        for i in range(self.num_particles_x):
            for j in range(self.num_particles_y):
                if i > 0:  # We can go west
                    self.swarm[i][j].neighbourhood.append(self.swarm[i - 1][j])
                if i < self.num_particles_x - 1:  # We can go east
                    self.swarm[i][j].neighbourhood.append(self.swarm[i + 1][j])
                if j > 0:  # We can go north
                    self.swarm[i][j].neighbourhood.append(self.swarm[i][j - 1])
                if j < self.num_particles_y - 1:  # We can go south
                    self.swarm[i][j].neighbourhood.append(self.swarm[i][j + 1])

    def loopOverParticles(self):
        # Loop over particles
        for i, row in enumerate(self.swarm):
            for j, col in enumerate(row):
                # Set weights according to mlpy particle
                self.nn.setAllWeights(self.swarm[i][j].position)

                result = self.nn.fire(np.array(self.batch_training_input))
                difference = self.batch_training_target - result
                error = np.mean(np.square(difference))

                self.swarm[i][j].error = error

                # Get & set personal best
                self.swarm[i][j].getPersonalBest()

        self.group_best_position = None
        self.group_best_error = float('inf')
        for i, row in enumerate(self.swarm):
            for j, col in enumerate(row):
                particle = self.swarm[i][j]
                neighbourhoodBest = particle.error
                neighbourhoodBestPos = particle.position

                for neighbour in particle.neighbourhood:
                    if abs(neighbour.error) < abs(neighbourhoodBest):
                        neighbourhoodBestPos = np.array(neighbour.position)
                        neighbourhoodBest = neighbour.error

                # Get current global best for this iteration
                if abs(particle.error) < abs(self.group_best_error):
                    self.group_best_position = np.array(particle.position)
                    self.group_best_error = particle.error

                # Get overall global best as well
                if abs(particle.error) < abs(self.best_error):
                    self.best_position = np.array(particle.position)
                    self.best_error = particle.error

                self.swarm[i][j].update_velocity(neighbourhoodBestPos)
                self.swarm[i][j].update_position(self.vmax)

