from rng import RNG
import math
import numpy as np

class Particle(object):
    def __init__(self, bounds, ng, weight, cognitiveConstant, socialConstant):
        self.position = None  # particle position
        self.velocity = None  # particle velocity

        self.best_position = None  # best position individual
        self.best_error = float('inf')  # best error individual

        self.error = None  # error individual

        self.num_dimensions = None
        self.ng = ng
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant
        self.bounds = bounds

    def initPos(self, pos):
        self.num_dimensions = len(pos)

        lo = 0.1 * self.bounds.minBound
        hi = 0.1 * self.bounds.maxBound

        self.velocity = (hi - lo) * np.random.random(self.num_dimensions) + lo
        self.position = np.array(pos)

    def getPersonalBest(self):
        if abs(self.error) < abs(self.best_error):
            self.best_position = np.array(self.position)
            self.best_error = self.error

        return self.best_error

    def update_velocity(self, group_best_position):
        assert not isinstance(self.ng.random(),
                              tuple), "You need to use a random generator that does not return a tuple"

        r1 = np.random.random(self.num_dimensions)
        r2 = np.random.random(self.num_dimensions)

        vel_cognitive = self.cognitiveConstant * r1 * (self.best_position - self.position)
        vel_social = self.socialConstant * r2 * (group_best_position - self.position)
        vel_inertia = self.weight * self.velocity
        self.velocity = vel_inertia + vel_cognitive + vel_social

        return self.velocity

    def update_position(self):
        # Clip the velocity so jumps are not too big Vmax
        # self.velocity = np.clip(self.velocity, -1, 1)

        # Update the position according to the velocity
        self.position = self.position + self.velocity

        # Clip the position to within the bounds
        self.position = np.clip(self.position, self.bounds.minBound, self.bounds.maxBound)

        # print(self.velocity)
        return self.velocity


    def toString(self):
        return ('\tPosition: {position}\n'+
                '\tBest Position: {pos_best}\n' +
                '\tError: {err}\n').format(position=self.position,
                                        pos_best=self.best_position,
                                        err=self.error)
