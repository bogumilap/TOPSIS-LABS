from typing import Callable

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BestSolutionSelection
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from follow_best_mutation import FollowBestGA
from follow_best_distinct import FollowBestDistinctGA
from repel_worst_gravity import RepelWorstGravity
from repel_worst_gravity_multistep import RepelWorstGravityMultistep
from labs import LABS


def run_genetic_algorithm(
        genetic_algorithm_class: Callable, mutation_probability: float,
        max_evaluations: int = 10000
):
    problem = LABS(100)
    algorithm = genetic_algorithm_class(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(mutation_probability),
        crossover=SPXCrossover(0.9),
        selection=BestSolutionSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
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

    return result


if __name__ == "__main__":
    run_genetic_algorithm(GeneticAlgorithm, 1.0)
    run_genetic_algorithm(FollowBestGA, 0.5)
    run_genetic_algorithm(FollowBestDistinctGA, 0.5)
    run_genetic_algorithm(RepelWorstGravity, 0.5)
    run_genetic_algorithm(RepelWorstGravityMultistep, 0.5)
