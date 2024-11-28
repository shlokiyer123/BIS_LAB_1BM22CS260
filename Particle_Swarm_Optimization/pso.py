import numpy as np

class Particle:
    def __init__(self, n_dimensions, minx, maxx):
        # Initialize particle's position and velocity
        self.position = np.random.uniform(minx, maxx, n_dimensions)
        self.velocity = np.random.uniform(-1, 1, n_dimensions)
        self.bestPos = np.copy(self.position)
        self.bestFitness = float('inf')  # Initialize to a large value
        self.fitness = float('inf')  # Fitness value will be updated later

    def evaluate(self, fitness_func):
        # Evaluate fitness of the current position
        self.fitness = fitness_func(self.position)

        # If current fitness is better than the best, update the best position
        if self.fitness < self.bestFitness:
            self.bestFitness = self.fitness
            self.bestPos = np.copy(self.position)

def pso(fitness_func, n_dimensions, N, max_iter, minx, maxx, w=0.5, c1=1.5, c2=1.5):
    # Initialize swarm (N particles)
    swarm = [Particle(n_dimensions, minx, maxx) for _ in range(N)]

    # Initialize the global best position and fitness
    best_fitness_swarm = float('inf')
    best_pos_swarm = np.zeros(n_dimensions)

    # Main PSO loop
    for gen in range(max_iter):
        avg_particle_best_fitness = 0  # Track the average best fitness of all particles

        for i in range(N):
            # Calculate new velocity
            r1, r2 = np.random.rand(2)
            swarm[i].velocity = (w * swarm[i].velocity +
                                 r1 * c1 * (swarm[i].bestPos - swarm[i].position) +
                                 r2 * c2 * (best_pos_swarm - swarm[i].position))

            # Update position
            swarm[i].position += swarm[i].velocity

            # Clip position to stay within bounds [minx, maxx]
            swarm[i].position = np.clip(swarm[i].position, minx, maxx)

            # Evaluate fitness and update personal best if necessary
            swarm[i].evaluate(fitness_func)

            # Update global best position if necessary
            if swarm[i].fitness < best_fitness_swarm:
                best_fitness_swarm = swarm[i].fitness
                best_pos_swarm = np.copy(swarm[i].position)

            # Accumulate the particle's best fitness for calculating average
            avg_particle_best_fitness += swarm[i].bestFitness

        # Calculate the average best fitness of all particles
        avg_particle_best_fitness /= N

        # Print progress (optional, can be commented out if not needed)
       # print(f"Generation {gen + 1}: Best Fitness = {best_fitness_swarm}, Avg Best Fitness = {avg_particle_best_fitness}")

    # Return the best position found by the swarm and other metrics
    return best_pos_swarm, best_fitness_swarm, avg_particle_best_fitness, max_iter

# Example: Rastrigin function (a common benchmark in optimization)
def rastrigin_function(x):
    n = len(x)
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# PSO parameters
n_dimensions = 2  # Example: 2D search space
N = 100            # Number of particles
max_iter = 100    # Number of iterations
minx = -100      # Minimum bound for position (for Rastrigin)
maxx = 100       # Maximum bound for position (for Rastrigin)

# Run the PSO algorithm with the Rastrigin function
best_position, best_fitness, avg_best_fitness, num_generations = pso(rastrigin_function, n_dimensions, N, max_iter, minx, maxx)

# Output the final results
print(f"\nGlobal Best Position: {best_position}")
print(f"Global Best Fitness Value: {best_fitness}")
print(f"Average Particle Best Fitness Value: {avg_best_fitness}")
print(f"Number of Generations: {num_generations}")
