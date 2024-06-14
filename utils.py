from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion


class ParametrizedGeneticAlgorithm(GeneticAlgorithm):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        solution_comparator: Comparator = ObjectiveComparator(0),
        mutate_one_gene: bool = False,
        N: int = 5,
    ):
        super(ParametrizedGeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_generator=population_generator,
            population_evaluator=population_evaluator,
            solution_comparator=solution_comparator,
        )
        self.mutate_one_gene = mutate_one_gene
        # number of best/worst individuals from which we choose the teacher/repeller
        ParametrizedGeneticAlgorithm.N = N
