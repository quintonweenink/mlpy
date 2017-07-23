import math

import numberGenerator
from particleSwarmOptimization.cpso import CPSO
from particleSwarmOptimization.structure.bounds import Bounds


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



nam_dimensions = 1
num_particles = 15
maxiter = 30
weight = 0.5  # constant inertia weight (how much to weigh the previous velocity)
cognitiveConstant = 1
socialConstant = 2
numberGenerator = numberGenerator.Lozi()

standardPSO = CPSO(func1, nam_dimensions, bounds, numberGenerator,
                   num_particles, maxiter, weight, cognitiveConstant, socialConstant)
standardPSO.establishSwarm()

standardPSO.begin()

