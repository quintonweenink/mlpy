import math

from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.bounds import Bounds

bounds1 = Bounds(-10, 10)
def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

bounds2 = Bounds(-10, 10)
def func2(x):
    return math.cos(x[0])



initial = [5, 5]  # initial starting location [x1,x2...]
num_particles = 15
maxiter = 30
standardPSO  = PSO(func2, initial, bounds2, num_particles, maxiter)
standardPSO.establishSwarm()

standardPSO.begin()

