"""
ABC Optimization Class

Implementation of Combinatorial Artificial Bee Colony (CABC), Reference:
Karaboga, D. & Gorkemli, B.
A combinatorial Artificial Bee Colony algorithm for traveling salesman problem 
2011 International Symposium on Innovations in Intelligent Systems and Applications, 2011, 50-53
"""

from pathlib import Path

import numpy as np
from numpy.random import default_rng

from timeit import default_timer as timer

from optimisation_base import OptimisationBase
from food_source import FoodSource
from candidate_solution import weight, inverse_mutation

from enum import Enum

import itertools

import math


class ABC_Optimisation(OptimisationBase):
    """ABC Optimization Class

    Parameters:
        p_reconnection              probability of reconnection in GSTM
        p_correction_perturation    probability of perturbation in GSTM
        p_linearity                 linearity of correction function in GSTM
    """

    # Initialization and built-in function overriding
    def __init__(self, parameters, output_path):

        super().__init__(parameters, output_path)

        self.p_reconnection = parameters["reconnection_probabilty"]  # 0.5
        self.p_correction_perturbation = parameters[
            "correction_perturbation_probability"
        ]  # 0.8
        self.p_linearity = parameters["linearity_probablity"]  # 0.2

        self.max_neighbourhood = parameters["max_neighbourhood"]  # 5
        self.limit = (self.size * self.problem.dimension) / parameters["L"]

        self.init_population(self.size)
        self.init_neighbourhoods()

    def __repr__(self):
        return f"<ABC Optimization size:{self.size} limit:{self.max_iterations} problem:{self.problem.name} >"

    def __str__(self):
        return "ABC Optimization: .. "

    def init_population(self, size):
        """Basic initialisation function for a random population of agents."""
        population = list()
        if self.heuristic_init:
            heuristic_tour = self.get_heuristic_tour()
            for _ in range(0, size):
                population.append(FoodSource(heuristic_tour))
        else:
            for _ in range(0, size):
                population.append(FoodSource())
        self.population = population

    def init_neighbourhoods(self):
        self.neighbourhoods = []
        for city in self.problem.get_nodes():
            neighbourhood = []
            for neighbour in self.problem.get_nodes():
                distance = self.problem.get_weight(city, neighbour)
                neighbourhood.append((neighbour, distance))
            neighbourhood = sorted(
                neighbourhood, key=lambda pair: pair[1], reverse=True
            )[: self.max_neighbourhood]
            self.neighbourhoods.append(
                list(neighbour[0] for neighbour in neighbourhood)
            )  # keep only the neighbour not the distance

    # Methods ------------------------------------------------------------------

    def run(
        self,
        print_states=True,
        print_joint_result=True,
        print_result_map=True,
        print_result_stats=True
    ):
        """
        Run the optimisation.
        Default settings will not print state each iteration, but only
        the result as both map, stats and combined output, as png and pgf.
        """

        rng = default_rng()
        start = timer()

        # Helper functions -----------------------------------------------------

        def d(a, b):
            return self.problem.get_weight(a, b)

        def gain(R1, R1_next, N1, N1_next):
            return d(R1, R1_next) + d(N1, N1_next) - d(R1, N1) + d(R1_next, N1_next)

        def choose_subtour(tourA, tourB):
            # Alternative Subtour Generation Function - Currently not in use!
            # Subtour generation as described in: Karaboga, D. & Gorkemli, B.,
            # Solving Traveling Salesman Problem by Using Combinatorial Artificial Bee Colony Algorithms,
            # International Journal on Artificial Intelligence Tools, 2019, 28, 1950004

            pos = rng.choice(
                range(1, self.problem.dimension)
            )  # careful: tsp instances have no city 0
            # print('pos: ' + str(pos))
            pos_index = tourA.index(pos)
            tourA = tourA[pos_index:] + tourA[:pos_index]  # pos_index = 0

            phi = rng.choice(range(-1, 1))

            if phi == -1:
                neighbour_inB = tourB[(tourB.index(pos) - 1) %
                                      self.problem.dimension]
                neighbour_index_inA = tourA.index(neighbour_inB)
                # print('neighbour' + str(neighbour_inB))

                open_tour = tourA[: neighbour_index_inA + 1]
                closed_tour = tourA[neighbour_index_inA + 1:]
                return closed_tour, open_tour

            else:
                neighbour_inB = tourB[(tourB.index(pos) + 1) %
                                      self.problem.dimension]

                neighbour_index_inA = tourA.index(neighbour_inB)

                open_tour = tourA[: neighbour_index_inA + 1]
                closed_tour = tourA[neighbour_index_inA + 1:]
                return open_tour, closed_tour

        def choose_random_subtour(tour):
            # Subtour generation as described in: Albayrak, M. & Allahverdi, N.
            # Development A New Mutation Operator to Solve the Traveling Salesman Problem by Aid of Genetic Algorithms
            # Expert Syst. Appl., 2011, 38, 1313-1320

            index = rng.choice(range(0, self.problem.dimension - 1))
            tour = tour[index:] + tour[:index]

            # Karaboga2019 uses dimension/2 instead of sqrt(dimension) as used by Albayrak
            # step = rng.choice(range(1, int(math.sqrt(self.problem.dimension))))
            step = rng.choice(range(1, int(self.problem.dimension / 2)))

            open_tour = tour[: step + 1]
            closed_tour = tour[step + 1:]
            return open_tour, closed_tour

        def reconnect(closed_tour, open_tour):
            min_tour = list(
                itertools.chain([closed_tour[0]], open_tour, closed_tour[1:])
            )
            min_weight = weight(self.problem, min_tour)
            for b in range(2, len(closed_tour)):
                new_tour = list(
                    itertools.chain(
                        closed_tour[:b], open_tour, closed_tour[b:])
                )
                new_weight = weight(self.problem, new_tour)
                if new_weight < min_weight:
                    min_tour = new_tour
            return min_tour

        def perturb(open_tour, closed_tour):
            new_tour = list(closed_tour)
            while open_tour:
                rnd = rng.random()
                if rnd <= self.p_linearity:
                    # Roll
                    rnd = rng.choice(range(0, len(open_tour)))
                    new_tour.append(open_tour.pop(rnd))
                else:
                    # Mix
                    new_tour.append(open_tour.pop(-1))
            return new_tour

        def correct(open_tour, closed_tour):

            dim = self.problem.dimension
            tour = closed_tour + open_tour
            segment_length = len(open_tour)

            # Endpoint Indices, Values and original Neighbours

            R1 = open_tour[0]
            R1_index = self.problem.dimension - segment_length
            R1_next_index = (R1_index + 1) % dim
            R1_next = tour[R1_next_index]
            R1_neighbours = (tour[R1_index - 1], R1_next)

            R2 = open_tour[-1]
            R2_index = dim - 1
            R2_next_index = (R2_index + 1) % dim
            R2_next = tour[R2_next_index]
            R2_neighbours = (tour[R2_index - 1], R2_next)

            # Neighbourhood Lists for Endpoints of the segment

            NL_R1 = self.neighbourhoods[
                R1 - 1
            ]  # caution - tsplib cities names 1...n, no ZERO
            for e in R1_neighbours:
                if e in NL_R1:
                    NL_R1.remove(e)

            NL_R2 = list(self.neighbourhoods[R2 - 1])
            for e in R2_neighbours:
                if e in NL_R2:
                    NL_R2.remove(e)

            new_tour = []
            inverted = False

            while not inverted and NL_R1:
                # Try to find neighbourhood swap for R1

                N1 = rng.choice(NL_R1)
                NL_R1.remove(N1)
                N1_index = tour.index(N1)
                N1_next_index = (N1_index + 1) % dim
                N1_next = tour[N1_next_index]

                if gain(R1, R1_next, N1, N1_next) > 0:
                    # invert segment [R1 + 1, N1] so that new edges (R1, N1), (R1 + 1, N1 + 1) are added
                    step = None
                    if R1_next_index > N1_index:
                        step = (dim - R1_next_index) + N1_index
                    else:
                        step = N1_index - R1_next_index
                    new_tour = inverse_mutation(tour, R1_next_index, step)
                    inverted = True

            while not inverted and NL_R2:
                # Try to find neighbourhood swap for R1

                N2 = rng.choice(NL_R2)
                NL_R2.remove(N2)
                N2_index = tour.index(N2)
                N2_next_index = (N2_index + 1) % dim
                N2_next = tour[N2_next_index]

                if gain(R2, R2_next, N2, N2_next) > 0:
                    # invert segment [R2 + 1, N2] so that new edges (R2, N2), (R2 + 1, N2 + 1) are added
                    step = None
                    if R2_next_index > N2_index:
                        step = (dim - R2_next_index) + N2_index
                    else:
                        step = N2_index - R2_next_index
                    new_tour = inverse_mutation(tour, R2_next_index, step)
                    inverted = True

            return new_tour

        def generate_candidate(current):
            self.run_stats["calls_generate_candidate"] += 1

            while True:
                other = rng.choice(self.population)
                if other != current:
                    break

            new_tour = []

            rnd_RC = rng.random()
            rnd_PC = rng.random()

            # open_tour, closed_tour = choose_subtour(current.tour, other.tour)
            closed_tour, open_tour = choose_random_subtour(current.tour)

            if rnd_RC <= self.p_reconnection:
                new_tour = reconnect(open_tour, closed_tour)
            elif rnd_PC <= self.p_correction_perturbation:
                new_tour = perturb(open_tour, closed_tour)
            else:
                new_tour = correct(open_tour, closed_tour)

            return FoodSource(new_tour)

        # Main Part of run() ---------------------------------------------------

        # TODO: probably belongs to init_population method not run()
        # Print Inital State and its best candidate (Iteration 0)
        best_agent = min(self.population, key=lambda p: p.weight)
        self.memory.append(best_agent)
        self.quality_by_iteration.append(best_agent.weight)
        self.quality_overall.append(
            min(self.memory, key=lambda p: p.weight).weight)

        self.run_stats.update(
            [
                ("min", best_agent.weight),
                ("iterations", 0),
                ("calls_generate_candidate", 0),
                ("calls_scout", 0),
            ]
        )

        # print intital state
        if print_states:
            self.print_state(self.population)

        # Iterate and Improve ----------------------------------
        continue_search = True
        while continue_search and self.iteration < self.max_iterations + 1:
            self.iteration += 1

            # cmd info output
            print(
                f"\rIteration {self.iteration:<6} of {self.max_iterations}", end="\r")

            # Employed Bee Phase ----------------------

            for food_source in self.population:
                candidate = generate_candidate(food_source)
                food_source.replace_with(candidate)

            # Onlooker Bee Phase ----------------------

            # for each "onlooker bee", aka. self.size times, do local search
            # according to probability distributing (depending on "fitness" of
            # food sources in population)

            p_distr = []
            for i in range(0, self.size):
                p_distr.append(self.population[i].fitness)
            best_fitness = max(p_distr)
            p_distr = list(0.9 * (p / best_fitness) + 0.1 for p in p_distr)
            # turn into probabiltiy distribution where sum is ONE
            accumulated_p = sum(p_distr)
            p_distr = list(p / accumulated_p for p in p_distr)

            for i in range(0, self.size):
                choice = np.random.choice(self.population, p=p_distr)
                candidate = generate_candidate(choice)

                choice.replace_with(candidate)

            # Scout Phase -----------------------------

            for food_source in self.population:
                if food_source.counter >= self.limit:
                    self.run_stats["calls_scout"] += 1
                    food_source.scout()

            # Save and output iteration results----------

            best_agent = min(self.population, key=lambda p: p.weight)

            # if it's a new minimum, save to stats
            self.memory.append(best_agent)
            if best_agent.weight < self.run_stats["min"]:
                self.run_stats["min"] = best_agent.weight
                self.run_stats["iterations"] = self.iteration

            self.quality_by_iteration.append(best_agent.weight)
            self.quality_overall.append(
                min(self.memory, key=lambda p: p.weight).weight)

            # print state of optimisation
            if print_states:
                self.print_state(self.population)

            # stop search if optimum is known and has been reached
            if self.optimum is not None:
                if self.run_stats["min"] == self.optimum.weight:
                    continue_search = False

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
