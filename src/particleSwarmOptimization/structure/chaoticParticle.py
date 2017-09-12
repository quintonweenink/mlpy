from particleSwarmOptimization.structure.particle import Particle
from numberGenerator.chaos.cprng import CPRNG


class ChaoticParticle(Particle):

    def update_velocity(self, group_best_position):
        r = self.ng.random()
        assert isinstance(r, tuple), "You need to use a chaotic pseudo random number generator (CPRNG) that returns a tuple"

        for i, velocity in enumerate(self.velocity):
            r = self.ng.random()

            r1, r2 = r[0], r[1]

            vel_cognitive = self.cognitiveConstant * r1 * (self.best_position[i] - self.position[i])
            vel_social = self.socialConstant * r2 * (group_best_position[i] - self.position[i])
            self.velocity[i] = self.weight * self.velocity[i] + vel_cognitive + vel_social
