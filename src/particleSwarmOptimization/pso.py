from particleSwarmOptimization.structure.particle import Particle


class PSO(object):

    def __init__(self, bounds, numberGenerator, num_particles, weight, cognitiveConstant, socialConstant):
        self.num_dimensions = None
        self.err_best_g = float('inf')                   # best error for group
        self.pos_best_g = None                  # best position for group

        self.err_best_i = float('inf')
        self.pos_best_i = None

        self.num_particles = num_particles
        self.bounds = bounds
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant

        self.swarm = []

    def getGlobalBest(self):
        self.err_best_i = float('inf')
        for j in range(self.num_particles):
            if abs(self.swarm[j].err_i) < abs(self.err_best_g):
                self.pos_best_g = self.swarm[j].position_i
                self.err_best_g = self.swarm[j].err_i
            # Get current best as well
            if abs(self.swarm[j].err_i) < abs(self.err_best_i):
                self.pos_best_i = self.swarm[j].position_i
                self.err_best_i = self.swarm[j].err_i

        return self.err_best_g

    def __str__(self):
        return "Best pos: " + str(self.pos_best_g) + "\nBest error: " + str(self.err_best_g) + "\n"
