import random

from numberGenerator.rng import RNG


class Particle(object):
    def __init__(self, bounds, ng, num_dimensions, costFunc,
                 weight, cognitiveConstant, socialConstant):
        self._position_i = []  # particle position
        self._velocity_i = []  # particle velocity
        self._pos_best_i = []  # best position individual
        self._err_best_i = -1  # best error individual
        self._err_i = -1  # error individual

        self._num_dimensions = num_dimensions
        self._ng = ng
        self._weight = weight
        self._cognitiveConstant = cognitiveConstant
        self._socialConstant = socialConstant
        self._costFunc = costFunc
        self._bounds = bounds

    def initPos(self):
        rng = RNG()
        for i in range(0, self._num_dimensions):
            self._velocity_i.append(rng.uniform(-1, 1))
            self.position_i.append(rng.uniform(self._bounds.minBound, self._bounds.maxBound))

    def evaluate(self):
        self._err_i = self._costFunc(self._position_i)

        # check to see if the current position is an individual best
        if self._err_i < self._err_best_i or self._err_best_i == -1:
            self._pos_best_i = self._position_i
            self._err_best_i = self._err_i


    def update_velocity(self, pos_best_g):
        for i in range(0, self._num_dimensions):
            r1 = self._ng.random()
            r2 = self._ng.random()

            vel_cognitive = self.cognitiveConstant * r1 * (self._pos_best_i[i] - self._position_i[i])
            vel_social = self.socialConstant * r2 * (pos_best_g[i] - self._position_i[i])
            self._velocity_i[i] = self.weight * self._velocity_i[i] + vel_cognitive + vel_social


    def update_position(self):
        for i in range(0, self._num_dimensions):
            self._position_i[i] = self.position_i[i] + self._velocity_i[i]

            # adjust maximum position if necessary
            if self._position_i[i] > self._bounds.maxBound:
                self._position_i[i] = self._bounds.maxBound

            # adjust minimum position if neseccary
            if self._position_i[i] < self._bounds.minBound:
                self._position_i[i] = self._bounds.minBound


    def toString(self):
        return ('\tPosition: {position}\n'+
                '\tBest Position: {pos_best}\n' +
                '\tError: {err}\n').format(position=self._position_i,
                                        pos_best=self._pos_best_i,
                                        err=self._err_i)

    @property
    def num_dimensions(self):
        return self._num_dimensions

    @num_dimensions.setter
    def num_dimensions(self, value):
        self._num_dimensions = value

    @property
    def ng(self):
        return self._ng

    @ng.setter
    def ng(self, value):
        self._ng = value

    @property
    def position_i(self):
        return self._position_i

    @position_i.setter
    def position_i(self, value):
        self._position_i = value

    @property
    def costFunc(self):
        return self._costFunc

    @costFunc.setter
    def costFunc(self, value):
        self._costFunc = value

    @property
    def err_i(self):
        return self._err_i

    @err_i.setter
    def err_i(self, value):
        self._err_i = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def cognitiveConstant(self):
        return self._cognitiveConstant

    @cognitiveConstant.setter
    def cognitiveConstant(self, value):
        self._cognitiveConstant = value

    @property
    def socialConstant(self):
        return self._weight

    @socialConstant.setter
    def socialConstant(self, value):
        self._socialConstant = value
