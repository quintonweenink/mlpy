import numpy as np

class PSO(object):

    def __init__(self, bounds, num_particles, weight, cognitiveConstant, socialConstant):
        self.num_dimensions = None

        self.group_best_error = float('inf')                   # best error for group
        self.group_best_position = None                  # best position for group

        self.best_error = float('inf')
        self.best_position = None

        self.num_particles = num_particles
        self.bounds = bounds
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant

        self.swarm = []

    def getGlobalBest(self):
        self.best_error = float('inf')
        for particle in self.swarm:
            if abs(particle.error) < abs(self.group_best_error):
                self.group_best_position = np.array(particle.position)
                self.group_best_error = particle.error
            # Get current best as well
            if abs(particle.error) < abs(self.best_error):
                self.best_position = np.array(particle.position)
                self.best_error = particle.error

        return self.group_best_position

    def __str__(self):
        return "Best pos: " + str(self.group_best_position) + "\nBest error: " + str(self.group_best_error) + "\n"
