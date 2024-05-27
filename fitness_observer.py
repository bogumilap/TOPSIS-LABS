from jmetal.core.observer import Observer


class FitnessObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, the best fitness and the computing time.
        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.fitness = []

    def update(self, *args, **kwargs):
        solutions = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]

        if (evaluations % self.display_frequency) == 0:
            self.fitness.append(solutions.objectives[0])

