from particleSwarmOptimization.pso import PSO

def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

initial = [5, 5]  # initial starting location [x1,x2...]
bounds = [(-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
num_particles = 15
maxiter = 30
standardPSO  = PSO(func1, initial, bounds, num_particles, maxiter)
standardPSO.establishSwarm()

standardPSO.begin()

