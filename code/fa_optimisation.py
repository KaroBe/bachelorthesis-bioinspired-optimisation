"""
FA Optimization Class

Implementation of Discrete Firefly Algorithm, Reference:
Jati, G. K.; Manurung, R. & Suyanto, Yang, X.-S.; Cui, Z.; Xiao, R.; Gandomi, A. H. & Karamanoglu, M. (Eds.)
Discrete Firefly Algorithm for Traveling Salesman Problem: A New Movement Scheme,
Swarm Intelligence and Bio-Inspired Computation, Elsevier, 2013, 295 - 312
"""

# ----------------------------------------------------------

from pathlib import Path

from numpy.random import default_rng

from timeit import default_timer as timer

from optimisation_base import OptimisationBase
from firefly import Firefly

# ------------------------------------------------------------------------------


class FA_Optimisation(OptimisationBase):
    """Allows optimisation state tracking and control.

    Parameters:
        m       maximum number of generated solutions in inversion mutation
        gamma   light absorption coefficient
    """

    # Initialization and built-in function overriding
    def __init__(self, parameters, output_path):

        super().__init__(parameters, output_path)

        self.m = parameters["m"]
        self.gamma = parameters["gamma"]

        self.init_population(self.size)

    def __repr__(self):
        return "<FA Optimization size:%s limit:%s problem:%s m:%s gamma:%s>" % (
            self.size,
            self.max_iterations,
            self.problem.name,
            self.m,
            self.gamma,
        )

    def __str__(self):
        return (
            "FA Optimization: population size %s, limit %s, problem %s, m:%s gamma:%s"
            % (self.size, self.max_iterations, self.problem.name, self.m, self.gamma)
        )

    def init_population(self, size):
        """Basic initialisation function for a random population of agents."""
        population = list()
        if self.heuristic_init:
            heuristic_tour = self.get_heuristic_tour()
            for _ in range(0, size):
                population.append(Firefly(heuristic_tour))
        else:
            for _ in range(0, size):
                population.append(Firefly())
        self.population = population

    # Methods ------------------------------------------------------------------

    def run(
        self,
        print_states=True,
        print_joint_result=True,
        print_result_map=True,
        print_result_stats=True,
        use_mutation_schedule=False,  # mutation_schedule not implemented
    ):
        """
        Run the optimisation.
        Default settings will not print state each iteration, but only
        the result as both map, stats and combined output, as png and pgf.
        """

        rng = default_rng()
        start = timer()

        def mutation_normal():
            new_agents = []
            for _ in range(0, self.m):
                # generate random starting index and random length step
                step = rng.choice(range(2, len(current_agent.tour)))
                index = rng.choice(range(0, len(current_agent.tour)))
                new_agents.append(
                    Firefly(current_agent.inverse_mutation(index, step)))
            # add to population and remove overall worst m agents
            self.population.extend(new_agents)
            self.population = sorted(self.population, key=lambda agent: agent.weight)[
                : self.size
            ]

        def mutation_schedule():
            # TODO: Mutation Schedule
            pass

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
                ("calls_move_towards", 0),
                ("calls_move_random", 0),
            ]
        )

        # print intital state
        if print_states:
            self.print_state(self.population)

        mutation_func = mutation_normal
        if use_mutation_schedule == True:
            mutation_func = mutation_schedule
            self.schedule = 1

        # Iterate and Improve
        continue_search = True
        while continue_search and self.iteration < self.max_iterations + 1:
            self.iteration += 1

            # cmd info
            print(
                f"\rIteration {self.iteration:<6} of {self.max_iterations}", end="\r")

            for current_agent in self.population:
                # print(f'Current Agent: {current_agent}')
                attractor = current_agent.find_brightest(
                    self.population, self.gamma)
                # there is no brighter agent (only equal ones, or self)
                # if distance(attractor, current_agent) == 0.0:
                if attractor.distance(current_agent) == 0.0:
                    mutation_func()
                    self.run_stats["calls_move_random"] += 1
                # there is a brighter agent
                else:
                    current_agent.move_towards(attractor)
                    self.run_stats["calls_move_towards"] += 1

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

            # self.print_map_only()

        end = timer()

        print(f'\rdone {"":<20}', end="\r")

        # Output Results ----------------------------------

        if print_joint_result:
            self.print_best()
        if print_result_map:
            self.print_map_only()
        if print_result_stats:
            self.print_stats_only()

        self.run_stats["time"] = end - start
        return (self.run_stats, self.quality_by_iteration, self.quality_overall)
