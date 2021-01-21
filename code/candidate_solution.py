"""
Base Class for Candidate Solutions

Holds a TSP tour and provides functions for tour manipulation 
and tour properties access.
"""


# ----------------------------------------------------------

import tsplib95 as tsp

import numpy as np
from numpy.random import default_rng

import itertools

# ---------------------------------------------------------


class CandidateSolution:
    """Base class for Candidate Solutions.

    Before use, the set_problem() function needs to be called on the class to
    set the problem for which agents (which represent candidate solutions) 
    are created!

    Attributes:
        tour    candidate solution associated with agent
        weight  weight of the tour
    """

    # Problem that candidate solutions are initialized on
    problem = tsp.models.StandardProblem()

    @classmethod
    def set_problem(self, problem):
        CandidateSolution.problem = problem

    # Constructor -------------------------------------------

    def __init__(self, tour=None):
        """ Initialize Candidate Solution
            tour not specified -> initialization with random tour
            tour specified -> initialization with given tour
        """
        if not tour:
            self.set_tour(random_tour(self.problem))
        else:
            self.set_tour(tour)

    def __repr__(self):
        return "<CandidateSolution tour:%s weight:%s>" % (self.tour, self.weight)

    def __str__(self):
        return "CandidateSolution: tour is %s, weight is %s" % (self.tour, self.weight)

    def set_tour(self, tour):
        self.tour = tour
        self.weight = weight(self.problem, self.tour)

    # Methods -----------------------------------------------

    def move_towards(self, attractor):
        """Move self towards attractor according to Jatis movement scheme."""
        # tools used by this method -------------------------

        def x_before_y(segment_xy, segment_i, segment_o):  # A
            return list(itertools.chain(segment_i, segment_xy, segment_o))

        def y_before_x(segment_xy, segment_i, segment_o):  # B
            return list(itertools.chain(segment_xy, segment_i, segment_o))

        def x_after_y(segment_xy, segment_i, segment_o):  # C
            return list(itertools.chain(segment_i, reversed(segment_xy), segment_o))

        def y_after_x(segment_xy, segment_i, segment_o):  # D
            return list(itertools.chain(reversed(segment_xy), segment_i, segment_o))

        movement = {
            0: x_before_y,  # A
            1: y_before_x,  # B
            2: x_after_y,  # C
            3: y_after_x,  # D
        }

        # test whether the input is correct

        if self.distance(attractor) == 0.0:
            print("move_towards was called on equal CandidateSolution")
            raise ValueError

        rng = default_rng()

        # new movement scheme --------------------------------

        new_tour = []

        edges_self = edge_set(self.tour)
        edges_attractor = edge_set(attractor.tour)
        delta_edges = edges_attractor.difference(edges_self)

        # get random edge [x,y] from attractor to be
        # incorporated into self
        pick_edge = rng.choice(range(0, len(delta_edges)))

        [x, y] = list(delta_edges)[pick_edge]

        # we want x to be the "left" and y to be the "right" node
        # rotate so that index_y = 0
        index_y = self.tour.index(y)
        self.set_tour(self.tour[index_y:] + self.tour[:index_y])

        # get indices of x and y
        index_y = 0
        index_x = self.tour.index(x)

        # get begin of x and y segments
        begin_y = index_y
        begin_x = index_x

        # find y - segment [begin_y:end_y+1]
        # walk to the right from begin_y = 0
        shared_edge = True
        end_y = begin_y
        while shared_edge and end_y + 1 != begin_x:
            # next edge which potentially expands segment
            next_edge = frozenset((self.tour[end_y], self.tour[end_y + 1]))
            # if the next edge is shared, continue, else stop at that
            if next_edge in edges_attractor:
                end_y += 1
            else:
                shared_edge = False
        segment_y = self.tour[0: end_y + 1]

        # find x - segment [end_x:begin_x+1]
        # walk to the left from begin_x = index(x)
        shared_edge = True
        end_x = begin_x
        while shared_edge and end_x - 1 != end_y:
            # next edge which potentially expands segment
            next_edge = frozenset((self.tour[end_x], self.tour[end_x - 1]))
            # if the next edge is shared, continue, else stop at that
            if next_edge in edges_attractor:
                end_x -= 1
            else:
                shared_edge = False
        segment_x = self.tour[end_x: begin_x + 1]

        # the segment in between x and y
        segment_i = self.tour[begin_x + 1:]

        # the segment in between end_y and end_x (edges not shared with
        # attractor), this may be [] if there is no such segment, which would
        # also be returned by the slice correctly (if it calls tour[i:i])
        segment_o = self.tour[end_y + 1: end_x]

        # combine segments for x and y
        segment_xy = segment_x + segment_y

        # do one of four merge options
        pick_move = rng.choice(range(0, 4))
        new_tour = movement[pick_move](segment_xy, segment_i, segment_o)

        # update values of CandidateSolution
        self.set_tour(new_tour)

    def inverse_mutation(self, index, step):
        """Return new tour based on inverse mutation of sequence [index,step] of tour of self."""
        return inverse_mutation(self.tour, index, step)

    def distance(self, other):
        """Return the distance between self and other.

        Returns distance as number of different edges
        """
        edges_self = edge_set(self.tour)
        edges_other = edge_set(other.tour)
        # A = number of different edges
        return len(edges_self.difference(edges_other))


# ----------------------------------------------------------


def weight(problem, tour):
    """Returns weight of tour according to problem"""
    # tours = []
    # tours.append(tour)
    weights = problem.trace_tours([tour])
    return weights[0]


def partial_weight(problem, partial_tour):
    """Return weight of partial tour, excluding the edge from last to first city."""
    weight = 0
    for i in range(1, len(partial_tour)):
        weight += problem.get_weight(partial_tour[i], partial_tour[i - 1])
    return weight


def random_tour(problem):
    """Returns randomly shuffled list of nodes"""
    # make a random number generator
    rng = default_rng()
    tour = list(problem.get_nodes())
    rng.shuffle(tour)
    return tour


def edge_set(tour):
    """return list of edges of the given tour"""
    starts = tour
    ends = list(tour)
    ends += [ends.pop(0)]
    edges = set()
    tuple_edges = zip(starts, ends)
    for edge in tuple_edges:
        edges.add(frozenset(edge))
    return edges


def inverse_mutation(tour, index, step):
    """
    Return new tour based on inverse mutation of sequence
    [index,step] of tour of self
    """
    inverse_tour = []
    # check wether section wraps around
    if index + step > len(tour):
        inverse_tour = tour[index:] + tour[:index]
        index = 0
    else:
        inverse_tour = list(tour)
    start = index
    end = index + step + 1
    inverse_tour[start:end] = reversed(inverse_tour[start:end])
    return inverse_tour
