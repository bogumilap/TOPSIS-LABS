from typing import Callable, Optional, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.observer import Observer
from jmetal.operator import BestSolutionSelection
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from follow_best import FollowBestGA
from follow_best_distinct import FollowBestDistinctGA
from repel_worst_gravity import RepelWorstGravity
from repel_worst_gravity_multistep import RepelWorstGravityMultistep

from combo_distinct_gravity import ComboDistinctGravity
from combo_distinct_gravity_multistep import ComboDistinctGravityMultistep
from combo_best_gravity import ComboBestGravity
from combo_best_gravity_multistep import ComboBestGravityMultistep

from labs import LABS


def run_genetic_algorithm(
    genetic_algorithm_class: Callable,
    mutation_probability: float,
    max_evaluations: int = 10000,
    observers: Optional[List[Observer]] = None,
):
    problem = LABS(50)
    algorithm = genetic_algorithm_class(
        problem=problem,
        population_size=20,
        offspring_population_size=10,
        mutation=BitFlipMutation(mutation_probability),
        crossover=SPXCrossover(0.5),
        selection=BestSolutionSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    # algorithm.observable.register(observer=PrintObjectivesObserver(1000))
    if observers is not None:
        for observer in observers:
            algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    print("Algorithm: {}".format(algorithm.get_name()))
    # print("Problem: {}".format(problem.name()))
    # print("Solution: {}".format(result.variables[0]))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
    print("===============================================")

    return result


if __name__ == "__main__":
    run_genetic_algorithm(GeneticAlgorithm, 0.5)

    run_genetic_algorithm(FollowBestGA, 0.5)
    run_genetic_algorithm(FollowBestDistinctGA, 0.5)
    run_genetic_algorithm(RepelWorstGravity, 0.5)
    run_genetic_algorithm(RepelWorstGravityMultistep, 0.5)

    run_genetic_algorithm(ComboDistinctGravity, 0.5)
    run_genetic_algorithm(ComboDistinctGravityMultistep, 0.5)
    run_genetic_algorithm(ComboBestGravity, 0.5)

    run_genetic_algorithm(ComboBestGravityMultistep, 0.5)
