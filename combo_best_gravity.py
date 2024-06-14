import random

from typing import List

from utils import ParametrizedGeneticAlgorithm
from jmetal.algorithm.singleobjective.genetic_algorithm import S


class ComboBestGravity(ParametrizedGeneticAlgorithm):
    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception("Wrong number of parents")

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)

                # "Combo Best Gravity" mutation
                if len(offspring_population) > 0:
                    # Follow part
                    # get top individuals in current population
                    best_individuals = self.solutions[: ComboBestGravity.N]
                    # randomly select one to be a teacher
                    teacher = best_individuals[
                        random.randint(0, len(best_individuals) - 1)
                    ]
                    # with given probability assign teacher's genes to current offspring's
                    if self.mutate_one_gene:
                        gene_index = random.randint(0, len(solution.variables[0]) - 1)
                        solution.variables[0][gene_index] = teacher.variables[0][
                            gene_index
                        ]
                    else:
                        for k in range(solution.number_of_variables):
                            for l in range(len(solution.variables[k])):
                                rand = random.random()
                                if rand <= self.mutation_operator.probability:
                                    solution.variables[k][l] = teacher.variables[k][l]

                    # Repel part
                    # get worst individuals in current population
                    worst_individuals = self.solutions[-ComboBestGravity.N :]

                    # randomly select one to be a repeller
                    repeller = worst_individuals[
                        random.randint(0, len(worst_individuals) - 1)
                    ]
                    # with given probability make current offspring's genes a negation of repeller's
                    if self.mutate_one_gene:
                        gene_index = random.randint(0, len(solution.variables[0]) - 1)
                        solution.variables[0][gene_index] = not repeller.variables[0][
                            gene_index
                        ]
                    else:
                        for k in range(solution.number_of_variables):
                            for l in range(len(solution.variables[k])):
                                rand = random.random()
                                if rand <= self.mutation_operator.probability:
                                    solution.variables[k][l] = not repeller.variables[
                                        k
                                    ][l]

                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_name(self) -> str:
        return "GA with 'Combo Best Gravity' mutation"
