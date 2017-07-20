import random

import numberGenerator
from particleSwarmOptimization.particle import Particle

class ChaoticParticle(Particle):
    def __init__(self, bounds, num_dimensions, costFunc, weight, cognitiveConstant, socialConstant):
        super(ChaoticParticle, self).__init__(bounds, num_dimensions, costFunc, weight, cognitiveConstant, socialConstant)
        self.ng = numberGenerator.CPRNG()