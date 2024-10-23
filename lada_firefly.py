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


def firefly_operator(
    w_fly,
    b_fly,
    *,
    attractiveness=2.0,
    absorption=1.0,
    levy_flight_param=2.0,
    random_param=None,
    t=1,
):
    """
    Improve the worse solution based on the better solution

    :param w_fly: the worse firefly/solution
    :param b_fly: the better firefly/solution

    :param attractiveness:    beta_0 in the FA operator
    :param absorption:        gamma in the FA operator
    :param levy_flight_param: lambda in the FA operator
    :param random_param:      alpha in the FA operator
    :param t:                 t in the FA operator

    :return: improved w_fly
    """

    random_param = random_param or random()  # use random() if not defined
    r2 = sum((b - w) ** 2 for w, b in zip(w_fly, b_fly))

    second_partial = attractiveness * np.exp(-absorption * r2)
    second_term = [second_partial * (b - w) for w, b in zip(w_fly, b_fly)]

    third_partial = random_param * np.sign(random() - 0.5)
    third_term = [third_partial * i * t ** (-levy_flight_param) for i in w_fly]

    return [a + b + c for a, b, c in zip(w_fly, second_term, third_term)]


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

    # use a DataFrame for displaying
    data = []
    for fitness, soln in zip(fitnesses, population):
        data.append([*soln, fitness])

    col_labels = [*(str(i) for i in range(1, dimension + 1)), "fitness"]
    df = pd.DataFrame(data, columns=col_labels)

    hr()
    print("Unsorted Population: ")
    print(df)

    hr()

    w_fly, b_fly = minmax_firefly(fitnesses)
    print("Worst and Best Firefly")
    print(df.iloc[[w_fly, b_fly]])

    hr()
    print("Applied operator on the worse firefly")
    improved_w_fly1 = firefly_operator(
        population[w_fly],
        population[b_fly],
        levy_flight_param=2,
    )
    improved_w_fly1.append(griewank(improved_w_fly1))

    improved_w_fly2 = firefly_operator(
        population[w_fly],
        population[b_fly],
        levy_flight_param=10,
    )
    improved_w_fly2.append(griewank(improved_w_fly2))

    print(
        pd.DataFrame(
            [improved_w_fly1, improved_w_fly2],
            columns=col_labels,
            index=["Setting 1", "Setting 2"],
        )
    )
    print("\n")


if __name__ == "__main__":
    main()
