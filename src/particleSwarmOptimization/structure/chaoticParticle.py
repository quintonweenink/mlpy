from particleSwarmOptimization.structure.particle import Particle


class ChaoticParticle(Particle):

    def update_velocity(self, pos_best_g):
        for i in range(self.num_dimensions):
            r = self.ng.random()

            assert isinstance(r, tuple), "You need to use a chaotic pseudo random number generator (CPRNG) that returns a tuple"

            r1, r2 = r[0], r[1]

            vel_cognitive = self.cognitiveConstant * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = self.socialConstant * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = self.weight * self.velocity_i[i] + vel_cognitive + vel_social
