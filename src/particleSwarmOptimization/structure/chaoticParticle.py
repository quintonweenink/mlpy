from particleSwarmOptimization.structure.particle import Particle
from numberGenerator.chaos.cprng import CPRNG


class ChaoticParticle(Particle):

    def __init__(self, bounds, numberGenerator, weight, cognitiveConstant, socialConstant):
        super(ChaoticParticle, self).__init__(bounds, weight, cognitiveConstant, socialConstant)
        self.numberGenerator = numberGenerator

    def update_velocity(self, group_best_position):
        r = self.numberGenerator.random()
        assert isinstance(self.numberGenerator, CPRNG), "You need to use a chaotic pseudo random number generator (CPRNG) that returns a tuple"

        for i, velocity in enumerate(self.position):
            r1, r2 = self.numberGenerator.random()

            vel_cognitive = self.cognitiveConstant * r1 * (self.best_position[i] - self.position[i])
            vel_social = self.socialConstant * r2 * (group_best_position[i] - self.position[i])
            self.velocity[i] = self.weight * velocity + vel_cognitive + vel_social
