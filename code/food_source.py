"""
FoodSource Class for ABC

Implements properties and functions of the candidate solutions in ABC.
"""

# ----------------------------------------------------------

from candidate_solution import CandidateSolution, random_tour, weight

import numpy as np

# ---------------------------------------------------------


class FoodSource(CandidateSolution):
    """
    FoodSource class for ABC optimisation.
    """

    # Constructor ---------------------------------------------------

    def __init__(self, tour=None):
        if not tour:
            self.set_tour(random_tour(self.problem))
        else:
            self.set_tour(tour)
        self.counter = 0

    def __repr__(self):
        return f'FoodSource tour:{self.tour} weight:{self.weight}'

    def __str__(self):
        return f'FoodSource: tour is {self.tour}, weight is {self.weight}'

    def set_tour(self, tour):
        self.tour = tour
        self.weight = weight(self.problem, self.tour)
        self.fitness = 1 / (1 + self.weight)

    # Special Methods -----------------------------------------------

    def replace_with(self, other):
        # if other tour is better than self, "move" by replacing fireflies
        # tour with other tour
        if other.weight < self.weight:
            self.set_tour(other.tour)
            self.counter = 0
        else:
            self.counter += 1

    def scout(self):
        self.set_tour(random_tour(self.problem))
        self.set_counter = 0
