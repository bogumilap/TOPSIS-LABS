from jmetal.core.observer import Observer

from benchmarks import ackley_benchmark, de_jong_benchmark, griewank_benchmark, rastrigin_benchmark


class BenchmarksObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, the best fitness and the computing time.
        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.benchmark_ackley = []
        self.benchmark_de_jong = []
        self.benchmark_rastrigin = []
        self.benchmark_griewank = []

    def update(self, *args, **kwargs):
        solutions = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]
        variables = solutions.variables[0]

        if (evaluations % self.display_frequency) == 0:
            self.benchmark_ackley.append(ackley_benchmark(variables))
            self.benchmark_de_jong.append(de_jong_benchmark(variables))
            self.benchmark_rastrigin.append(rastrigin_benchmark(variables))
            self.benchmark_griewank.append(griewank_benchmark(variables))

