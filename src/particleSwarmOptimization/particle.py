import random

import numberGenerator


class Particle(object):
    def __init__(self, bounds, num_dimensions, costFunc, weight, cognitiveConstant, socialConstant):
        self.__position_i = []  # particle position
        self.__velocity_i = []  # particle velocity
        self.__pos_best_i = []  # best position individual
        self.__err_best_i = -1  # best error individual
        self.__err_i = -1  # error individual
        self.__num_dimensions = 2
        self.__ng = numberGenerator.RNG()
        self.__weight = weight
        self.__cognitiveConstant = cognitiveConstant
        self.__socialConstant = socialConstant
        self.__costFunc = costFunc
        self.__bounds = bounds

        for i in range(0, num_dimensions):
            self.__num_dimensions = num_dimensions
            self.__velocity_i.append(self.__ng.uniform(-1, 1))
            self.position_i.append(self.__ng.uniform(self.__bounds.minBound, self.__bounds.maxBound))

    # evaluate current fitness
    def evaluate(self):
        self.__err_i = self.__costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.__err_i < self.__err_best_i or self.__err_best_i == -1:
            self.__pos_best_i = self.position_i
            self.__err_best_i = self.__err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        for i in range(0, self.__num_dimensions):
            r1 = self.__ng.random()
            r2 = self.__ng.random()

            vel_cognitive = self.cognitiveConstant * r1 * (self.__pos_best_i[i] - self.position_i[i])
            vel_social = self.socialConstant * r2 * (pos_best_g[i] - self.position_i[i])
            self.__velocity_i[i] = self.weight * self.__velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0, self.__num_dimensions):
            self.position_i[i] = self.position_i[i] + self.__velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > self.__bounds.maxBound:
                self.position_i[i] = self.__bounds.maxBound

            # adjust minimum position if neseccary
            if self.position_i[i] < self.__bounds.minBound:
                self.position_i[i] = self.__bounds.minBound

    def toString(self):
        return ('\tPosition: {position}\n'+
                '\tBest Position: {pos_best}\n' +
                '\tError: {err}\n').format(position=self.position_i,
                                        pos_best=self.__pos_best_i,
                                        err=self.err_i)

    @property
    def position_i(self):
        return self.__position_i

    @position_i.setter
    def position_i(self, value):
        self.__position_i = value

    @property
    def ng(self):
        return self.__ng

    @ng.setter
    def ng(self, value):
        self.__ng = value

    @property
    def costFunc(self):
        return self.__costFunc

    @costFunc.setter
    def costFunc(self, value):
        self.__costFunc = value

    @property
    def err_i(self):
        return self.__err_i

    @err_i.setter
    def err_i(self, value):
        self.__err_i = value

    @property
    def weight(self):
        return self.__weight

    @weight.setter
    def weight(self, value):
        self.__weight = value

    @property
    def cognitiveConstant(self):
        return self.__cognitiveConstant

    @cognitiveConstant.setter
    def cognitiveConstant(self, value):
        self.__cognitiveConstant = value

    @property
    def socialConstant(self):
        return self.__weight

    @socialConstant.setter
    def socialConstant(self, value):
        self.__socialConstant = value