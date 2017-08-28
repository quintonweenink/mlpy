from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.structure.chaoticParticle import ChaoticParticle


class CPSO(PSO):

    def establishSwarm(self):
        for i in range(self.num_particles):
            self.swarm.append(ChaoticParticle(self.bounds, self.ng, self.num_dimensions, self.costFunc, self.weight, self.cognitiveConstant, self.socialConstant))
            self.swarm[i].initPos()