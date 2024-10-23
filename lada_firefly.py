import numpy as np
from random import random
from numpy import cos, sqrt


def griewank(dimensions):
    first_term, second_term = 0, 1

    # split the function for readability
    # we calculate both terms in one pass
    for i, x in enumerate(dimensions):
        first_term += x * x / 4000
        second_term *= cos(x) / sqrt(i + 1)

    return first_term + second_term + 1


def bounded_generate(lb, ub):
    return lb + (ub - lb) * random()


bounds = -10, 10

population, pop_size = [], 10
dimension = 5
fitnesses = []

for k in range(pop_size):
    solution = [bounded_generate(*bounds) for _ in range(dimension)]
    population.append(solution)

    fitnesses.append(griewank(solution))

# display values
for i in range(pop_size):
    print(f"Solution {i}: {population[i]}")

for i in range(pop_size):
    print(f"Fitness {i}: {fitnesses[i]:.5f}")
