from particleSwarmOptimization.structure.particle import Particle


class PSO(object):

    def __init__(self, costFunc, num_dimensions, bounds, numberGenerator,
                 num_particles, maxiter, weight, cognitiveConstant, socialConstant):
        self.num_dimensions = num_dimensions
        self.err_best_g = float('inf')                   # best error for group
        self.pos_best_g = []                   # best position for group

        self.err_best_i = float('inf')
        self.pos_best_i = []

        self.maxiter = maxiter
        self.num_particles = num_particles
        self.bounds = bounds
        self.ng = numberGenerator
        self.costFunc = costFunc
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant

        self.swarm = []

    def establishSwarm(self):
        for i in range(self.num_particles):
            self.swarm.append(Particle(self.bounds, self.ng, self.num_dimensions, self.costFunc, self.weight, self.cognitiveConstant, self.socialConstant))
            self.swarm[i].initPos()

    def evaluateSwarm(self):
        for j in range(self.num_particles):
            self.swarm[j].evaluate()

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

    def begin(self):
        for i in range(self.maxiter):
            self.evaluateSwarm()
            self.getGlobalBest()

            for j in range(self.num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position()
            print(self)

        print('FINAL:')
        print(self)


    def __str__(self):
        return "Best pos: " + str(self.pos_best_g) + "\nBest error: " + str(self.err_best_g) + "\n"
