import math

from particleSwarmOptimization.pso import PSO
from particleSwarmOptimization.bounds import Bounds

bounds = Bounds(-10, 10)


def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

def func2(x):
    return math.cos(x[0])

def func3(x):
    return math.sin(x[0])



#initial = [5, 5]  # initial starting location [x1,x2...]
nam_dimensions = 1
num_particles = 15
maxiter = 30
weight = 0.5  # constant inertia weight (how much to weigh the previous velocity)
cognitiveConstant = 1
socialConstant = 2
standardPSO  = PSO(func3, nam_dimensions, bounds, num_particles, maxiter, weight, cognitiveConstant, socialConstant)
#standardPSO.costFunc(func3)
standardPSO.establishSwarm()

standardPSO.begin()

