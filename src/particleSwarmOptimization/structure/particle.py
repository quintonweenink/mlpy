from rng import RNG
import math
import numpy as np

class Particle(object):
    def __init__(self, bounds, ng, weight, cognitiveConstant, socialConstant):
        self.position_i = None  # particle position
        self.velocity_i = None  # particle velocity
        self.pos_best_i = None  # best position individual
        self.err_best_i = float('inf')  # best error individual
        self.err_i = None  # error individual

        self.num_dimensions = None
        self.ng = ng
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant
        self.bounds = bounds

    def initPos(self, pos):
        self.num_dimensions = len(pos)

        self.velocity_i = np.zeros(self.num_dimensions)
        self.position_i = pos

    # def evaluate(self):
    #     self.err_i = self.costFunc(self.position_i)
    #
    #     self.getPersonalBest()

    def getPersonalBest(self):
        if abs(self.err_i) < abs(self.err_best_i):
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

        return self.err_best_i

    def update_velocity(self, pos_best_g):
        assert not isinstance(self.ng.random(),
                              tuple), "You need to use a random generator that does not return a tuple"

        r1 = np.zeros(self.num_dimensions)
        r2 = np.zeros(self.num_dimensions)

        for i in range(self.num_dimensions):
            r1[i] = self.ng.random()
            r2[i] = self.ng.random()

        vel_cognitive = self.cognitiveConstant * r1 * (self.pos_best_i - self.position_i)
        vel_social = self.socialConstant * r2 * (pos_best_g - self.position_i)
        self.velocity_i = (self.weight * self.velocity_i) + vel_cognitive + vel_social


    def update_position(self):
        self.position_i = self.position_i + self.velocity_i

        self.position_i = np.clip(self.position_i, self.bounds.minBound, self.bounds.maxBound)


    def toString(self):
        return ('\tPosition: {position}\n'+
                '\tBest Position: {pos_best}\n' +
                '\tError: {err}\n').format(position=self.position_i,
                                        pos_best=self.pos_best_i,
                                        err=self.err_i)
