import math

from chaos.lozi import Lozi
from src.particleSwarmOptimization.cpso import CPSO
from src.particleSwarmOptimization.structure.bounds import Bounds


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
weight = 0.5
cognitiveConstant = 1
socialConstant = 2

numberGenerator = Lozi()

i = 0
while i < 1000:
    print(numberGenerator.random())
    i += 1


standardPSO = CPSO(func3, nam_dimensions, bounds, numberGenerator,
                   num_particles, maxiter, weight, cognitiveConstant, socialConstant)
standardPSO.establishSwarm()

standardPSO.begin()

