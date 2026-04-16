import numpy as np
import math

# Hàm mục tiêu Ackley
def cost_function(x):
    a = 20
    b = 0.2
    c = 2 * math.pi
    d = len(x)

    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return term1 + term2 + a + math.e


# Tham số PSO
DIMENSIONS = 2
POPULATION = 30
MAX_ITER = 100

MIN_BOUND = -5
MAX_BOUND = 5
V_MAX = 0.5

w = 0.7
c1 = 1.5
c2 = 1.5


class Particle:
    def __init__(self):
        self.position = np.random.uniform(MIN_BOUND, MAX_BOUND, DIMENSIONS)
        self.velocity = np.random.uniform(-V_MAX, V_MAX, DIMENSIONS)

        self.best_position = self.position.copy()
        self.best_fitness = cost_function(self.position)
        self.fitness = self.best_fitness


def particle_swarm_optimization():
    swarm = [Particle() for _ in range(POPULATION)]

    # Khởi tạo global best
    gbest_position = swarm[0].best_position.copy()
    gbest_fitness = swarm[0].best_fitness

    for particle in swarm:
        if particle.best_fitness < gbest_fitness:
            gbest_fitness = particle.best_fitness
            gbest_position = particle.best_position.copy()

    # Lặp tối ưu
    for iteration in range(MAX_ITER):
        for particle in swarm:
            r1 = np.random.rand(DIMENSIONS)
            r2 = np.random.rand(DIMENSIONS)

            # Cập nhật vận tốc
            particle.velocity = (
                w * particle.velocity
                + c1 * r1 * (particle.best_position - particle.position)
                + c2 * r2 * (gbest_position - particle.position)
            )

            # Giới hạn vận tốc
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)

            # Cập nhật vị trí
            particle.position = particle.position + particle.velocity

            # Giới hạn vị trí
            particle.position = np.clip(particle.position, MIN_BOUND, MAX_BOUND)

            # Tính fitness mới
            particle.fitness = cost_function(particle.position)

            # Cập nhật best cá nhân
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()

            # Cập nhật best toàn cục
            if particle.fitness < gbest_fitness:
                gbest_fitness = particle.fitness
                gbest_position = particle.position.copy()

        print(f"Iteration {iteration + 1}/{MAX_ITER}, Best Fitness = {gbest_fitness:.6f}")

    return gbest_position, gbest_fitness


if __name__ == "__main__":
    best_position, best_fitness = particle_swarm_optimization()

    print("\nOptimal Solution Found:")
    print("Best Position:", best_position)
    print("Best Fitness:", best_fitness)