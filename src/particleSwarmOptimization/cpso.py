from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.structure.chaoticParticle import ChaoticParticle


class CPSO(PSO):

    def establishSwarm(self):
        for i in range(0, self._num_particles):
            self._swarm.append(ChaoticParticle(self._bounds, self._ng, self._num_dimensions, self._costFunc, self._weight, self._cognitiveConstant, self._socialConstant))
            self._swarm[i].initPos()