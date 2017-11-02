import numpy as np
import math

from bounds import Bounds
from mlpy.particleSwarmOptimization.pso import PSO
from mlpy.particleSwarmOptimization.structure.particle import Particle

np.set_printoptions(suppress=True)

errors = []
bounds = Bounds(-10, 10)

# Create the mlpy with the nn weights
num_particles = 50
inertia_weight = 0.729
cognitiveConstant = 1.49
socialConstant = 1.49
num_dimensions = 50
# Configure PSO
pso = PSO(bounds, num_particles, inertia_weight, cognitiveConstant, socialConstant)

def error(position):
    err = 0.0
    for i in range(len(position)):
        xi = position[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return err

# Create particles
for i in range(pso.num_particles):
    pso.swarm.append(Particle(bounds, inertia_weight, cognitiveConstant, socialConstant))
    pso.swarm[i].initPos(4 * np.random.random(num_dimensions) - 2)

# Iterate over training data
for i in range(2000):
    # Loop over particles
    for j in range(pso.num_particles):

        # Fire the neural network and calculate error
        pso.swarm[j].error = error(pso.swarm[j].position)

        # Get & set personal best
        pso.swarm[j].getPersonalBest()

        # Print results
        print(j, np.array(pso.swarm[j].error))

    # Get & set global best
    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

    if(i % 1 == 0):
        print("Best error:\t\t\t" + str(pso.group_best_error))
        print("Current best error:\t" + str(pso.best_error) + "\n")

print('FINAL:')
print(pso)