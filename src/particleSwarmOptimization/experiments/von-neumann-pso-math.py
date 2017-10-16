import numpy as np
import math

from bounds import Bounds
from src.particleSwarmOptimization.pso import PSO
from src.particleSwarmOptimization.structure.particle import Particle

np.set_printoptions(suppress=True)

errors = []
bounds = Bounds(-10, 10)

# Create the pso with the nn weights
num_particles = 7
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
    row = []
    for j in range(pso.num_particles):
        particle = Particle(bounds, inertia_weight, cognitiveConstant, socialConstant)
        particle.initPos(4 * np.random.random(num_dimensions) - 2)
        row.append(particle)

    pso.swarm.append(row)

for i in range(pso.num_particles):
    for j in range(pso.num_particles):
        if i > 0: # We can go west
            pso.swarm[i][j].neighbourhood.append(pso.swarm[i - 1][j])
        if i < pso.num_particles - 1: # We can go east
            pso.swarm[i][j].neighbourhood.append(pso.swarm[i + 1][j])
        if j > 0: # We can go north
            pso.swarm[i][j].neighbourhood.append(pso.swarm[i][j - 1])
        if j < pso.num_particles - 1: # We can go south
            pso.swarm[i][j].neighbourhood.append(pso.swarm[i][j + 1])


# Iterate over training data
for x in range(2000):
    # Loop over particles
    for i, row in enumerate(pso.swarm):
        for j, col in enumerate(row):

            # Fire the neural network and calculate error
            pso.swarm[i][j].error = error(pso.swarm[i][j].position)

            # Get & set personal best
            pso.swarm[i][j].getPersonalBest()

            # Print results
            print(i, j, np.array(pso.swarm[i][j].error))

    for i in range(pso.num_particles):
        for j in range(pso.num_particles):
            particle = pso.swarm[i][j]
            neighbourhoodBest = particle.error
            neighbourhoodBestPos = particle.position

            for neighbour in particle.neighbourhood:
                if abs(neighbour.error) < abs(neighbourhoodBest):
                    neighbourhoodBestPos = np.array(neighbour.position)
                    neighbourhoodBest = neighbour.error
                # Get current global best as well
                if abs(neighbour.error) < abs(pso.best_error):
                    pso.best_position = np.array(particle.position)
                    pso.best_error = particle.error

            pso.swarm[i][j].update_velocity(neighbourhoodBestPos)
            pso.swarm[i][j].update_position()

    if(x % 1 == 0):
        print("Current best error:\t" + str(pso.best_error) + "\n")

print('FINAL:')
print(pso)