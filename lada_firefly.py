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

for k in range(pop_size):
    solution = [bounded_generate(*bounds)]
