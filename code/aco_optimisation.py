"""
All the classes needed for Ant Colony Optimisation (ACO).

Classes:
    ACO_Optimisation
    PheromoneMatrix
    HeuristicMatrix

Implementation of Ant Colony System, Reference:
Dorigo, M. & Gambardella, L. M.
Ant colony system: a cooperative learning approach to the traveling salesman problem 
IEEE Transactions on Evolutionary Computation, 1997, 1, 53-66 
"""

from pathlib import Path

import numpy as np
from numpy.random import default_rng

from timeit import default_timer as timer

from optimisation_base import OptimisationBase
from ant import Ant
from candidate_solution import CandidateSolution, weight, edge_set


class ACO_Optimisation(OptimisationBase):
    """Ant Colony Optimisation Class.

    Parameters:
        alpha       influence of pheromone
        beta        influence of heuristic information
        tau_zero    initial pheromone value

        phi         pheromone decay coefficient (0,1] - local (online) pheromone update
        rho         evaporation rate (0,1] - global (offline) pheromone update
        q_zero      probability of exploitative choice behaviour vs. explorative behaviour
       (q_const     constant in quantity of pheromone deposited: delta tau(k,i,j) = Q/L_k )
    """

    # Initialization and built-in function overriding
    def __init__(self, parameters, output_path):
        super().__init__(parameters, output_path)

        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]
        self.tau_zero = parameters["tau_zero"]
        self.phi = parameters["phi"]
        self.rho = parameters["rho"]
        self.q_zero = parameters["q_zero"]

        # This is actually only relevant for Ant System, not Ant Colony System
        # self.q_const = parameters['q_const']

        self.heuristic_tour = self.get_heuristic_tour()
        self.heuristic_weight = weight(self.problem, self.heuristic_tour)
        # self.tau_zero = 1 / (self.problem.dimension * self.heuristic_weight)

        # Initialisation of Pheromone Matrix and Heuristic Information
        self.pheromone_matrix = PheromoneMatrix(
            self.alpha, self.tau_zero, self.problem.dimension
        )
        self.heuristic_matrix = HeuristicMatrix(self.problem, self.beta)

    def __repr__(self):
        return f"<ACO Optimization size:{self.size} limit:{self.max_iterations} problem:{self.problem.name} >"

    def __str__(self):
        return "ABC Optimization: .. "

    def init_population(self):
        """Basic initialisation function for a new random population of agents."""
        self.population = []  # delete old population
        for _ in range(0, self.size):
            self.population.append(Ant())

    # Methods ------------------------------------------------------------------

    def run(
        self,
        print_states=True,
        print_joint_result=True,
        print_result_map=True,
        print_result_stats=True,
        offline_update="iteration",  # or 'overall'
    ):
        """
        Run the optimisation.
        Default settings will not print state each iteration, but only
        the result as both map, stats and combined output, as png and pgf.
        """

        rng = default_rng()
        start = timer()

        # Helper functions -----------------------------------------------------

        # solution component attractiveness
        def get_sc_attractiveness(ant):
            attr = {}
            for sc in ant.solution_components:
                attr_sc = self.pheromone_matrix.get_weighted_pheromone(
                    ant.partial_solution[-1], sc
                ) * self.heuristic_matrix.get_weighted_heuristic(
                    ant.partial_solution[-1], sc
                )
                attr[sc] = attr_sc
            return attr

        def construction_step(ant):
            self.run_stats["calls_construction_step"] += 1
            sc_attractiveness_dict = get_sc_attractiveness(ant)

            # Pseudorandom proportional rule
            random = rng.random()  # \in [0,1)
            if random <= self.q_zero:
                # add the most attractive edge
                ant.add_to_partial_solution(max(sc_attractiveness_dict))
            else:
                # add edge according to probability distribution
                probability_distribution = []
                total_attractiveness = sum(sc_attractiveness_dict.values())
                for sc in ant.solution_components:
                    probability_distribution.append(
                        sc_attractiveness_dict[sc] / total_attractiveness
                    )
                sc = rng.choice(ant.solution_components,
                                p=probability_distribution)
                ant.add_to_partial_solution(sc)

            # local pheromone update
            new_edge = ant.partial_solution[-2], ant.partial_solution[-1]
            new_tau = (1 - self.phi) * self.pheromone_matrix.get_pheromone(
                *new_edge
            ) + (self.phi * self.tau_zero)
            self.pheromone_matrix.set_pheromone(*new_edge, new_tau)

        # Main Part of run() ---------------------------------------------------

        # Random Population or Heuristic Population
        population = list()
        if self.heuristic_init:
            heuristic_tour = self.get_heuristic_tour()
            for _ in range(0, self.size):
                population.append(CandidateSolution(heuristic_tour))
        else:
            for _ in range(0, self.size):
                population.append(CandidateSolution())

        self.population = population

        # Initialize pheromone matrix with Heuristic Tour
        if self.heuristic_init:

            edges = edge_set(self.heuristic_tour)
            for edge in edges:
                edge = tuple(edge)
                delta_tau = 1 / self.heuristic_weight
                new_tau = (1 - self.rho) * self.pheromone_matrix.get_pheromone(
                    *edge
                ) + self.rho * delta_tau
                self.pheromone_matrix.set_pheromone(*edge, new_tau)

        # Save Iteration Zero
        best_agent = min(self.population, key=lambda p: p.weight)
        self.memory.append(best_agent)
        self.quality_by_iteration.append(best_agent.weight)
        self.quality_overall.append(
            min(self.memory, key=lambda p: p.weight).weight)

        self.run_stats.update(
            [
                ("min", best_agent.weight),
                ("iterations", 0),
                ("calls_construction_step", 0)
            ]
        )

        # print intital state
        if print_states:
            self.print_state(self.population)

        # Iterate and Improve ----------------------------------
        continue_search = True
        while continue_search and self.iteration < self.max_iterations + 1:
            self.iteration += 1

            # cmd info
            print(
                f"\rIteration {self.iteration:<6} of {self.max_iterations}", end="\r")

            # Search ----------------------

            # New population of ants
            self.init_population()

            # Construct Solutions
            for _ in range(1, self.problem.dimension):
                for ant in self.population:
                    construction_step(ant)  # updates ants

            # Optional Local Search could be done at this point

            # Save iteration results

            best_agent = min(self.population, key=lambda ant: ant.weight)
            self.memory.append(best_agent)

            # Offline Pheromone Update

            best_agent_overall = min(self.memory, key=lambda ant: ant.weight)
            update_agent = None
            if offline_update == "iteration":
                update_agent = best_agent
            else:
                # offline_update == 'overall'
                update_agent = best_agent_overall

            edges = edge_set(update_agent.tour)
            for edge in edges:
                edge = tuple(edge)
                delta_tau = 1 / update_agent.weight
                new_tau = (1 - self.rho) * self.pheromone_matrix.get_pheromone(
                    *edge
                ) + self.rho * delta_tau
                self.pheromone_matrix.set_pheromone(*edge, new_tau)

            # Save and output iteration results----------

            # if it's a new minimum, save to stats
            if best_agent.weight < self.run_stats["min"]:
                self.run_stats["min"] = best_agent.weight
                self.run_stats["iterations"] = self.iteration

            self.quality_by_iteration.append(best_agent.weight)
            self.quality_overall.append(best_agent_overall.weight)

            # print state of optimisation
            if print_states:
                self.print_state(self.population)

            # stop search if optimum is known and has been reached
            if self.optimum is not None:
                if self.run_stats["min"] == self.optimum.weight:
                    continue_search = False

            # print(self.pheromone_matrix.matrix)

        end = timer()

        # Output Results ----------------------------------

        print(f'\rdone {"":<20}', end="\r")

        if print_joint_result:
            self.print_best()
        if print_result_map:
            self.print_map_only()
        if print_result_stats:
            self.print_stats_only()

        self.run_stats["time"] = end - start
        return (self.run_stats, self.quality_by_iteration, self.quality_overall)


class PheromoneMatrix:
    """Class for Matrix of Pheromones for edges in tsp problem."""

    def __init__(self, alpha, tau_zero, dimension):
        self.alpha = alpha
        self.matrix = []
        for i in range(0, dimension):
            self.matrix.append([])
            for _ in range(0, dimension):
                self.matrix[i].append(tau_zero)
        # print(self.matrix)

    def get_pheromone(self, i, j):
        return self.matrix[i - 1][j - 1]

    def get_weighted_pheromone(self, i, j):
        return pow(self.matrix[i - 1][j - 1], self.alpha)

    def set_pheromone(self, i, j, val):
        self.matrix[i - 1][j - 1] = val
        self.matrix[j - 1][i - 1] = val


class HeuristicMatrix:
    """Class for Matrix of Heuristic Information."""

    def __init__(self, problem, beta):
        self.beta = beta
        self.matrix = []
        for i in range(0, problem.dimension):
            self.matrix.append([])
            for j in range(0, problem.dimension):
                if i == j:
                    self.matrix[i].append(0)
                else:
                    self.matrix[i].append(1 / problem.get_weight(i + 1, j + 1))

    def get_heuristic(self, i, j):
        return self.matrix[i - 1][j - 1]

    def get_weighted_heuristic(self, i, j):
        return pow(self.matrix[i - 1][j - 1], self.beta)

    def to_dict(self):
        # TODO: output the pheromone matrix to visualize it
        pass
