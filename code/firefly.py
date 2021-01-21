"""
Ant Class for FA

Implements properties and functions of the candidate solutions in FA
"""

# ----------------------------------------------------------

from candidate_solution import CandidateSolution

import numpy as np

# ---------------------------------------------------------


class Firefly(CandidateSolution):
    """
    Firefly Class for FA Optimisation.
    """

    # Constructor ---------------------------------------------------

    def __repr__(self):
        return "<Firefly tour:%s weight:%s brightness:%s>" % (self.tour, self.weight, self.brightness)

    def __str__(self):
        return "Firefly: tour is %s, weight is %s, brightness is %s" % (self.tour, self.weight, self.brightness)

    def set_tour(self, tour):
        super().set_tour(tour)
        self.brightness = 1/self.weight

    # Overwritten Methods -------------------------------------------

    def distance(self, other):
        """Returns distance as number of different edges, scaled to [0,10]"""
        # A = distance as number of different edges
        A = super().distance(other)
        # n = number of cities = number of edges in tour
        n = len(self.tour)
        # "distance" scaled in interval [0, 10]
        # return (A/n * 10) / n
        return (10 * A) / pow(n, 2)

    # Special Methods -----------------------------------------------

    def find_brightest(self, population, gamma):
        """
        Return the brightest neighbour relative to distance, including self
        """
        # print('~ Find brightest ~')
        # print('Brightness self: %s' % (self.brightness))
        attractor = Firefly()
        best_brightness = 0
        # find brightest CandidateSolution from own position
        for other in population:
            #print('Brightness neighbour: %s' % (other.brightness))
            new_brightness = self.relative_brightness(other, gamma)
            # if a better attractor is found it replaces previous
            #print('Relative: %s' % (new_brightness))
            if new_brightness > best_brightness:
                # print('replace!')
                best_brightness = new_brightness
                attractor = other
        # either self, or the firstly found best is returned
        #print('Best found: %s' % (attractor.brightness))
        return attractor

    def relative_brightness(self, other, gamma):
        """
        Returns brightness (attractiveness) of other candidate solution
        relative to self, dependent on distance and gamma
        """
        # r = distance(self, other)
        r = self.distance(other)
        exponent = - gamma * pow(r, 2)
        attr = other.brightness * np.exp(exponent)
        return attr
