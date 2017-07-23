from particleSwarmOptimization.structure.particle import Particle


class PSO(object):

    def __init__(self, costFunc, num_dimensions, bounds, numberGenerator,
                 num_particles, maxiter, weight, cognitiveConstant, socialConstant):
        self._num_dimensions = num_dimensions
        self._err_best_g = -1                   # best error for group
        self._pos_best_g = []                   # best position for group

        self._maxiter = maxiter
        self._num_particles = num_particles
        self._bounds = bounds
        self._ng = numberGenerator
        self._costFunc = costFunc
        self._weight = weight
        self._cognitiveConstant = cognitiveConstant
        self._socialConstant = socialConstant

        self._swarm = []

    def establishSwarm(self):
        for i in range(0, self._num_particles):
            self._swarm.append(Particle(self._bounds, self._ng, self._num_dimensions, self._costFunc, self._weight, self._cognitiveConstant, self._socialConstant))
            self._swarm[i].initPos()


    def setGlobalBest(self):
        print('[')
        for j in range(0, self._num_particles):
            self._swarm[j].evaluate()
            print(self._swarm[j].toString())

            # determine if current particle is the best (globally)
            if self._swarm[j].err_i < self._err_best_g or self._err_best_g == -1:
                self._pos_best_g = list(self._swarm[j].position_i)
                self._err_best_g = float(self._swarm[j].err_i)
        print(']')

    def begin(self):
        # begin optimization loop
        i=0
        while i < self._maxiter:
            self.setGlobalBest()

            # cycle through _swarm and update velocities and position
            for j in range(0,self._num_particles):
                self._swarm[j].update_velocity(self._pos_best_g)
                self._swarm[j].update_position()
            i+=1
            self.printGlobalBest()

        print('FINAL:')
        self.printGlobalBest()


    def printGlobalBest(self):
        print(self._pos_best_g)
        print(self._err_best_g)
