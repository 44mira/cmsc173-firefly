import numpy as np
import pandas as pd
from random import random
from numpy import cos, sqrt


# [[ Functions ]] {{{


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


def minmax_firefly(fitnesses):
    # Get worst and best firefly (this makes the algorithm O(n) instead of O(n log n))
    w_fly, b_fly = 0, 0
    for i, fitness in enumerate(fitnesses):
        if fitness <= fitnesses[b_fly]:
            b_fly = i
        if fitness >= fitnesses[w_fly]:
            w_fly = i

    return w_fly, b_fly


# }}}


def main():
    hr = lambda: print("-" * 32)

    bounds = -10, 10

    population, pop_size = [], 10
    dimension = 5
    fitnesses = []

    for _ in range(pop_size):
        solution = [bounded_generate(*bounds) for _ in range(dimension)]
        population.append(solution)

        fitnesses.append(griewank(solution))

    data = []
    for fitness, soln in zip(fitnesses, population):
        tmp = soln
        tmp.append(fitness)
        data.append(tmp)

    col_labels = [*(str(i) for i in range(1, dimension + 1)), "fitness"]
    df = pd.DataFrame(data=data, columns=col_labels)

    hr()
    print("Unsorted Population: ")
    print(df)

    hr()

    w_fly, b_fly = minmax_firefly(fitnesses)
    print("Best and Worst Firefly\n")
    print(df.iloc[[w_fly, b_fly]])


if __name__ == "__main__":
    main()
