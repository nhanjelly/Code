import numpy as np
import random

# ====== 1. Dữ liệu ======
distances = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
])

n = len(distances)

# ====== 2. Tính Lnn (Nearest Neighbor) ======
def nearest_neighbor():
    visited = [0]
    total = 0

    while len(visited) < n:
        current = visited[-1]
        next_city = min(
            [c for c in range(n) if c not in visited],
            key=lambda x: distances[current][x]
        )
        total += distances[current][next_city]
        visited.append(next_city)

    total += distances[visited[-1]][visited[0]]
    return total

Lnn = nearest_neighbor()

# ====== 3. Khởi tạo pheromone ======
tau0 = 1 / (n * Lnn)
pheromone = np.full((n, n), tau0)

# ====== 4. Tham số ======
n_ants = 10
n_iterations = 50
beta = 2
rho = 0.1       # global evaporation
phi = 0.1       # local update
q0 = 0.9

# ====== 5. Hàm tính khoảng cách ======
def calculate_distance(path):
    return sum(distances[path[i]][path[i+1]]
               for i in range(len(path)-1)) \
           + distances[path[-1]][path[0]]

# ====== 6. Biến toàn cục ======
best_path = None
best_distance = float("inf")

# ====== 7. ACS ======
for iteration in range(n_iterations):

    for ant in range(n_ants):
        visited = [random.randint(0, n - 1)]

        while len(visited) < n:
            current = visited[-1]
            unvisited = [c for c in range(n) if c not in visited]

            q = random.random()

            # ===== Exploitation =====
            if q <= q0:
                next_city = max(
                    unvisited,
                    key=lambda city: pheromone[current][city] *
                                     (1 / distances[current][city]) ** beta
                )

            # ===== Exploration =====
            else:
                probs = []
                for city in unvisited:
                    tau = pheromone[current][city]
                    eta = (1 / distances[current][city]) ** beta
                    probs.append(tau * eta)

                probs = np.array(probs)
                probs /= probs.sum()

                next_city = np.random.choice(unvisited, p=probs)

            visited.append(next_city)

            # ===== Local update =====
            pheromone[current][next_city] = \
                (1 - phi) * pheromone[current][next_city] + phi * tau0

        # ===== Đánh giá =====
        dist = calculate_distance(visited)

        if dist < best_distance:
            best_distance = dist
            best_path = visited

    # ===== Global update =====
    pheromone *= (1 - rho)

    for i in range(len(best_path) - 1):
        pheromone[best_path[i]][best_path[i+1]] += rho * (1 / best_distance)
        pheromone[best_path[i+1]][best_path[i]] += rho * (1 / best_distance)

    pheromone[best_path[-1]][best_path[0]] += rho * (1 / best_distance)
    pheromone[best_path[0]][best_path[-1]] += rho * (1 / best_distance)

# ====== 8. Kết quả ======
print("Best Path:", best_path)
print("Best Distance:", best_distance)