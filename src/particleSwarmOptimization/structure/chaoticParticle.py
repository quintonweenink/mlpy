from particleSwarmOptimization.structure.particle import Particle


class ChaoticParticle(Particle):

    def update_velocity(self, pos_best_g):
        for i in range(0, self.num_dimensions):
            r = self._ng.random()
            if not isinstance(r, tuple):
                exit("EXIT: You need to use a number generator that returns tuple")

            r1, r2 = r[0], r[1]

            vel_cognitive = self._cognitiveConstant * r1 * (self._pos_best_i[i] - self._position_i[i])
            vel_social = self._socialConstant * r2 * (pos_best_g[i] - self._position_i[i])
            self._velocity_i[i] = self._weight * self._velocity_i[i] + vel_cognitive + vel_social
