from typing import Callable

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BestSolutionSelection
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.comparator import ObjectiveComparator
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from follow_best_mutation import FollowBestGA
from labs import LABS


def run_genetic_algorithm(genetic_algorithm_class: Callable, mutation_probability: float):
    problem = LABS(10)
    solution_comparator = ObjectiveComparator(0)
    algorithm = genetic_algorithm_class(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(mutation_probability),
        crossover=PMXCrossover(0.9),
        selection=BestSolutionSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=1000),
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(1000))

    algorithm.run()
    result = algorithm.get_result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: {}".format(result.variables[0]))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
    print("===============================================")


if __name__ == "__main__":
    run_genetic_algorithm(GeneticAlgorithm, 1.0)
    run_genetic_algorithm(FollowBestGA, 0.5)
