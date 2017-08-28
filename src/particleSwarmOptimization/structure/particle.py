from rng import RNG


class Particle(object):
    def __init__(self, bounds, ng, num_dimensions, costFunc,
                 weight, cognitiveConstant, socialConstant):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        self.num_dimensions = num_dimensions
        self.ng = ng
        self.weight = weight
        self.cognitiveConstant = cognitiveConstant
        self.socialConstant = socialConstant
        self.costFunc = costFunc
        self.bounds = bounds

    def initPos(self):
        rng = RNG()
        for i in range(self.num_dimensions):
            self.velocity_i.append(rng.uniform(-1, 1))
            self.position_i.append(rng.uniform(self.bounds.minBound, self.bounds.maxBound))

    def evaluate(self):
        self.err_i = self.costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i


    def update_velocity(self, pos_best_g):
        for i in range(self.num_dimensions):
            r1 = self.ng.random()
            r2 = self.ng.random()

            vel_cognitive = self.cognitiveConstant * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = self.socialConstant * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = self.weight * self.velocity_i[i] + vel_cognitive + vel_social


    def update_position(self):
        for i in range(self.num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > self.bounds.maxBound:
                self.position_i[i] = self.bounds.maxBound

            # adjust minimum position if neseccary
            if self.position_i[i] < self.bounds.minBound:
                self.position_i[i] = self.bounds.minBound


    def toString(self):
        return ('\tPosition: {position}\n'+
                '\tBest Position: {pos_best}\n' +
                '\tError: {err}\n').format(position=self.position_i,
                                        pos_best=self.pos_best_i,
                                        err=self.err_i)
