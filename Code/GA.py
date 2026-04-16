import numpy as np
import copy
import matplotlib.pyplot as plt

# cost function
def sphere(x):
    return sum(x**2)
# Placeholder for every individual
population = {}
# population size
npop = 20
# number of variables
num_var = 5
# lower bound
varmin = -10
# upper bound
varmax = 10
# cost function
costfunc = sphere

# Mỗi cá thể đều có vị trí (nhiễm sắc thể) và chi phí.
for i in range(npop):
    population[i] = {'position': None, 'cost': None}

for i in range(npop):
    population[i]['position'] = np.random.uniform(varmin, varmax, num_var)
    population[i]['cost'] = costfunc(population[i]['position'])
# roulette wheel selection
def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
# Calculating probability for roulette wheel selection
beta = 1
costs = []

for i in range(len(population)):
    costs.append(population[i]['cost'])

costs = np.array(costs)

avg_cost = np.mean(costs)

if avg_cost != 0:
    costs = costs / avg_cost

probs = np.exp(-beta * costs)
# select parents
p1 = population[roulette_wheel_selection(probs)]
p2 = population[roulette_wheel_selection(probs)]
# crossover
def crossover(p1, p2):

    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)
    alpha = np.random.uniform(0, 1, c1['position'].shape)
    c1['position'] = alpha * p1['position'] + (1 - alpha) * p2['position']
    c2['position'] = alpha * p2['position'] + (1 - alpha) * p1['position']

    return c1, c2
# mutation
def mutate(c, mu, sigma):
    # mu - mutation rate
    # sigma - step size of mutation
    y = copy.deepcopy(c)
    flag = np.random.rand(*c['position'].shape) <= mu
    ind = np.argwhere(flag)
    y['position'][ind] += sigma * np.random.randn(*ind.shape)
    return y
# parameters
mu = 0.1
sigma = 0.1
# best solution
bestsol_cost = population[0]

for i in range(len(population)):
    if population[i]['cost'] < bestsol_cost['cost']:
        bestsol_cost = copy.deepcopy(population[i])
# crossover
c1, c2 = crossover(p1, p2)

# mutation
c1 = mutate(c1, mu, sigma)
c2 = mutate(c2, mu, sigma)

# Evaluate first off spring
c1['cost'] = costfunc(c1['position'])

if c1['cost'] < bestsol_cost['cost']:
    bestsol_cost = copy.deepcopy(c1)
# Evaluate second off spring
c2['cost'] = costfunc(c2['position'])

if c2['cost'] < bestsol_cost['cost']:
    bestsol_cost = copy.deepcopy(c2)

print("Best Cost:", bestsol_cost['cost'])
print("Best Position:", bestsol_cost['position'])
# plot result
plt.plot([population[i]['cost'] for i in range(npop)])
plt.title("Genetic Algorithm")
plt.xlabel("Population")
plt.ylabel("Cost")
plt.show()