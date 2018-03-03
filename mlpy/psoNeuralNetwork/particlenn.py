import numpy as np

from mlpy.particleSwarmOptimization.structure.particle import Particle


class ParticleNN(Particle):

    def getPersonalBest(self):
        if abs(self.error) < abs(self.best_error):
            self.best_position = np.array(self.position)
            self.best_error = self.error

        return self.best_error
