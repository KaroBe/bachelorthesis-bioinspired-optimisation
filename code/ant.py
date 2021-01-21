"""
Ant Class for ACO

Implements properties and functions of the candidate solutions in ACO
"""

from candidate_solution import CandidateSolution, random_tour, weight, partial_weight
from numpy.random import default_rng


class Ant(CandidateSolution):
    """
    Ant class for ACO optimisation.
    """

    # Constructor ---------------------------------------------------

    def __init__(self):
        self.rng = default_rng()
        self.reset()

    def __repr__(self):
        return f"Ant: partial tour:{self.partial_solution} weight:{self.weight} tour: {self.tour}, weight: {self.weight}"

    def __str__(self):
        return f"Ant: partial tour:{self.partial_solution} weight:{self.weight} tour: {self.tour}, weight: {self.weight}"

    # Special Methods -----------------------------------------------

    def reset(self):

        self.partial_solution = []
        self.partial_weight = 0
        self.solution_components = list(self.problem.get_nodes())
        self.tour = None
        self.weight = None

        self.add_to_partial_solution(self.rng.choice(self.solution_components))

    def add_to_partial_solution(self, solution_component):
        """
        Removes solution component from the list of available solution 
        component and adds it to the partial solution.
        """
        index = self.solution_components.index(solution_component)
        self.partial_solution.append(self.solution_components.pop(index))
        self.partial_weight = partial_weight(
            self.problem, self.partial_solution)

        # If the construction is complete, set tour and weight
        if not self.solution_components:
            self.set_tour(self.partial_solution)
