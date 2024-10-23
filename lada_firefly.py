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
    row_labels = [f"Soln {i}" for i in range(1, pop_size + 1)]
    df = pd.DataFrame(data=data, columns=col_labels, index=row_labels)

    hr()
    print("Unsorted Population: ")
    print(df)

    hr()
    print("Sorted Population: ")
    print(df.sort_values("fitness"))


if __name__ == "__main__":
    main()
