from math import cos, e, pi, prod, sqrt


def ackley_benchmark(x):
    a = 20
    b = 0.2
    c = 2 * pi
    n = len(x)
    return -a * e ** (-b * sqrt(sum((xi ** 2 for xi in x)) / n)) - e ** (sum((cos(c * xi) for xi in x)) / n) + a + e


def de_jong_benchmark(x):
    return sum((xi ** 2 for xi in x))


def rastrigin_benchmark(x):
    n = len(x)
    return 10 * n + sum((xi ** 2 - 10 * cos(2 * pi * xi) for xi in x))


def griewank_benchmark(x):
    return sum((xi ** 2 / 4000 for xi in x)) - prod((cos(xi / sqrt(i + 1)) for i, xi in enumerate(x))) + 1
