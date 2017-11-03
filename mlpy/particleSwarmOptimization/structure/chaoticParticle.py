from mlpy.numberGenerator.chaos.cprng import CPRNG
from mlpy.particleSwarmOptimization.structure.particle import Particle


class ChaoticParticle(Particle):

    def __init__(self, bounds, numberGenerator, weight, cognitiveConstant, socialConstant):
        super(ChaoticParticle, self).__init__(bounds, weight, cognitiveConstant, socialConstant)
        self.numberGenerator = numberGenerator

        assert isinstance(self.numberGenerator,
                          CPRNG), "You need to use a chaotic pseudo random number generator (CPRNG)"


    def update_velocity(self, group_best_position):
        size = len(self.position)

        r1 = self.numberGenerator.randomArray(size)
        r2 = self.numberGenerator.randomArray(size)

        vel_cognitive = self.cognitiveConstant * r1 * (self.best_position - self.position)
        vel_social = self.socialConstant * r2 * (group_best_position - self.position)
        vel_inertia = self.weight * self.velocity
        self.velocity = vel_inertia + vel_cognitive + vel_social

        return self.velocity
