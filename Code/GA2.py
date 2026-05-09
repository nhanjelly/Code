import numpy as np
import random
import math
import matplotlib.pyplot as plt

depot = (40, 50) 

Q = 200 

customers = [
    (45, 68, 10, 912, 967), (45, 70, 30, 825, 870), (42, 66, 10, 65, 146),
    (42, 68, 10, 727, 782), (42, 65, 10, 15, 67), (40, 69, 20, 621, 702),
    (40, 66, 20, 170, 225), (38, 68, 20, 255, 324), (38, 70, 10, 534, 605),
    (35, 66, 10, 357, 410), (35, 69, 10, 448, 505), (25, 85, 20, 652, 721),
    (22, 75, 30, 30, 92), (22, 85, 10, 567, 620), (20, 80, 40, 384, 429),
    (20, 85, 40, 475, 528), (18, 75, 20, 99, 148), (15, 75, 20, 179, 254),
    (15, 80, 10, 278, 345), (30, 50, 10, 10, 73), (30, 52, 20, 914, 965),
    (28, 52, 20, 812, 883), (28, 55, 10, 732, 777), (25, 50, 10, 65, 144),
    (25, 52, 40, 169, 224), (25, 55, 10, 622, 701), (23, 52, 10, 261, 316),
    (23, 55, 20, 546, 593), (20, 50, 10, 358, 405), (20, 55, 10, 449, 504),
    (10, 35, 20, 200, 237), (10, 40, 30, 31, 100), (8, 40, 40, 87, 158),
    (8, 45, 20, 751, 816), (5, 35, 10, 283, 344), (5, 45, 10, 665, 716),
    (2, 40, 20, 383, 434), (0, 40, 30, 479, 522), (0, 45, 20, 567, 624),
    (35, 30, 10, 264, 321), (35, 32, 10, 166, 235), (33, 32, 20, 68, 149),
    (33, 35, 10, 16, 80), (32, 30, 10, 359, 412), (30, 30, 10, 541, 600),
    (30, 32, 30, 448, 509), (30, 35, 10, 1054, 1127), (28, 30, 10, 632, 693),
    (28, 35, 10, 1001, 1066), (26, 32, 10, 815, 880), (25, 30, 10, 725, 786),
    (25, 35, 10, 912, 969), (44, 5, 20, 286, 347), (42, 10, 40, 186, 257),
    (42, 15, 10, 95, 158), (40, 5, 30, 385, 436), (40, 15, 40, 35, 87),
    (38, 5, 30, 471, 534), (38, 15, 10, 651, 740), (35, 5, 20, 562, 629),
    (50, 30, 10, 531, 610), (50, 35, 20, 262, 317), (50, 40, 50, 171, 218),
    (48, 30, 10, 632, 693), (48, 40, 10, 76, 129), (47, 35, 10, 826, 875),
    (47, 40, 10, 12, 77), (45, 30, 10, 734, 777), (45, 35, 10, 916, 969),
    (95, 30, 30, 387, 456), (95, 35, 20, 293, 360), (53, 30, 10, 450, 505),
    (92, 30, 10, 478, 551), (53, 35, 50, 353, 412), (45, 65, 20, 997, 1068),
    (90, 35, 10, 203, 260), (88, 30, 10, 574, 643), (88, 35, 20, 109, 170),
    (87, 30, 10, 668, 731), (85, 25, 10, 769, 820), (85, 35, 30, 47, 124),
    (75, 55, 20, 369, 420), (72, 55, 10, 265, 338), (70, 58, 20, 458, 523),
    (68, 60, 30, 555, 612), (66, 55, 10, 173, 238), (65, 55, 20, 85, 144),
    (65, 60, 30, 645, 708), (63, 58, 10, 737, 802), (60, 55, 10, 20, 84),
    (60, 60, 10, 836, 889), (67, 85, 20, 368, 441), (65, 85, 40, 475, 518),
    (65, 82, 10, 285, 336), (62, 80, 30, 196, 239), (60, 80, 10, 95, 156),
    (60, 85, 30, 561, 622), (58, 75, 20, 30, 84), (55, 80, 10, 743, 820),
    (55, 85, 20, 647, 726)
]

# =============================
# 2. HÀM FITNESS 
# =============================

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def fitness(route):
    total_dist = 0
    current = depot
    load = 0

    for i in route:
        cust = customers[i]

        if load + cust[2] > Q:
            total_dist += distance(current, depot)
            current = depot
            load = 0

        total_dist += distance(current, cust[:2])
        current = cust[:2]
        load += cust[2]

    total_dist += distance(current, depot)
    return total_dist


# =============================
# 3.
# =============================

def create_individual():
    route = list(range(len(customers)))
    random.shuffle(route)
    return route

def create_population(n):
    return [create_individual() for _ in range(n)]


# =============================
# 4. SELECTION (Tournament)
# =============================

def selection(pop):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if fitness(pop[i]) < fitness(pop[j]) else pop[j]


# =============================
# 5. CROSSOVER (OX)
# =============================

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    child = [-1]*size
    child[a:b] = p1[a:b]

    ptr = 0
    for x in p2:
        if x not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = x

    return child


# =============================
# 6. MUTATION (swap)
# =============================

def mutate(ind):
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind


# =============================
# 7. GA MAIN
# =============================

pop_size = 50
generations = 100
mutation_rate = 0.2

population = create_population(pop_size)

best = population[0]

for gen in range(generations):

    new_pop = []

    for _ in range(pop_size):
        p1 = selection(population)
        p2 = selection(population)

        child = crossover(p1, p2)

        if random.random() < mutation_rate:
            child = mutate(child)

        new_pop.append(child)

    population = new_pop

    # cập nhật best
    for ind in population:
        if fitness(ind) < fitness(best):
            best = ind

    print(f"Gen {gen} | Best = {fitness(best):.2f}")


# =============================
# 8. OUTPUT
# =============================

print("\nBest route:", best)
print("Best distance:", fitness(best))

def plot_route(route):
    x = []
    y = []

    current = depot

    plt.scatter(depot[0], depot[1], c='red', s=100, label='Depot')

    for i in route:
        cust = customers[i]

        plt.scatter(cust[0], cust[1], c='blue')

        plt.plot([current[0], cust[0]], [current[1], cust[1]], 'k-')

        current = cust[:2]

    plt.plot([current[0], depot[0]], [current[1], depot[1]], 'k-')

    plt.title("Best Route (GA)")
    plt.legend()
    plt.show()


plot_route(best)