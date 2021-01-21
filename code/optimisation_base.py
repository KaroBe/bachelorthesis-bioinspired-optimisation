"""
Base Class for Optimization Algorithms

Implements problem instance loading, heuristic initialization function, and data plotting functions.
"""

# ----------------------------------------------------------

import tsplib95 as tsp

from numpy.random import default_rng

# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


from pathlib import Path

from candidate_solution import CandidateSolution, weight

# from agent import distance

# ------------------------------------------------------------------------------


class OptimisationBase:
    """
    Abstract base class for Optimisation Algorithms
    """

    # Initialization and built-in function overriding
    def __init__(self, parameters, output_path):

        self.output_path = output_path
        self.size = parameters["size"]
        self.max_iterations = parameters["max_iterations"]
        self.heuristic_init = parameters["heuristic_init"]

        self.load_problem(parameters["problem_filename"])

        self.iteration = 0
        self.memory = []
        self.quality_by_iteration = []
        self.quality_overall = []
        self.run_stats = {**parameters}

    def __repr__(self):
        return "<Optimization size:%s limit:%s problem:%s>" % (
            self.size,
            self.max_iterations,
            self.problem.name,
        )

    def __str__(self):
        return "Optimization: population size %s, limit %s, problem %s" % (
            self.size,
            self.max_iterations,
            self.problem.name,
        )

    def get_heuristic_tour(self):
        def d(a, b):
            return self.problem.get_weight(a, b)

        rng = default_rng()
        nodes = list(self.problem.get_nodes())
        first = rng.choice(nodes)
        nodes.remove(first)
        tour = []
        tour.append(first)
        while nodes:
            next_node = min([(node, d(tour[-1], node)) for node in nodes])
            tour.append(next_node[0])
            nodes.remove(next_node[0])
        return tour

    def load_problem(self, problem_filename):
        """
        Load problem from file, as well as opt.tour if available
        """
        self.problem = tsp.load(problem_filename)
        CandidateSolution.set_problem(self.problem)
        self.optimum = None
        opt_tour = problem_filename[:-4] + ".opt.tour"
        try:
            self.optimum = CandidateSolution(tsp.load(opt_tour).tours[0])
        except FileNotFoundError as err:
            print("FileNotFoundError: {0}".format(err))
        else:
            pass

    # Output Methods -----------------------------------------------------------

    def print_best(self):

        # make figure, axes
        plt.style.use("ggplot")
        plt.tight_layout()

        gs_kw = dict(width_ratios=[3, 2], height_ratios=[1])
        fig, (ax1, ax2) = plt.subplots(
            figsize=(9, 4.5), ncols=2, nrows=1, gridspec_kw=gs_kw
        )

        ax1.set(title="Optimisation Result", xlabel="x", ylabel="y")
        # ax1.set_aspect('equal', 'box')
        ax1.set_aspect("equal")
        ax2.set(title="Quality development", xlabel="iteration", ylabel="tour length")

        # AX1 Best Solution vs. Optimal Solution

        # Nodes
        xs, ys = zip(*self.problem.node_coords.values())
        labels = self.problem.node_coords.keys()
        ax1.scatter(xs, ys, marker="o", color="dimgrey", zorder=10)
        for label, x, y in zip(labels, xs, ys):
            ax1.annotate(label, xy=(x, y), zorder=20)
        # xs,ys hold data for city coordinates
        xs, ys = zip(*self.problem.node_coords.values())
        labels = self.problem.node_coords.keys()

        # plots best tour in self.memory
        best_tour = min(self.memory, key=lambda p: p.weight).tour
        xt = []
        yt = []
        for p in best_tour:
            coords = self.problem.node_coords[p]
            xt.append(coords[0])
            yt.append(coords[1])
        xt.append(xt[0])
        yt.append(yt[0])
        ax1.plot(xt, yt, alpha=1.0, color="darkred", linestyle="dashed", zorder=2)

        # plots optimum tour if given
        if self.optimum is not None:
            opt_tour = self.optimum.tour
            xt = []
            yt = []
            for p in opt_tour:
                coords = self.problem.node_coords[p]
                xt.append(coords[0])
                yt.append(coords[1])
            xt.append(xt[0])
            yt.append(yt[0])
            ax1.plot(xt, yt, alpha=0.4, color="yellowgreen", linewidth=5, zorder=1)

        # Labels
        redline = mlines.Line2D(
            [], [], color="darkred", linestyle="dashed", label="Overall Best Tour"
        )
        yellowline = mlines.Line2D(
            [], [], color="yellowgreen", linewidth="5", label="Known Optimal Tour"
        )
        grey_dot = mlines.Line2D(
            [], [], color="dimgrey", marker="o", linestyle="", label="Node"
        )
        ax1.legend(
            handles=[redline, yellowline, grey_dot],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            shadow=True,
            ncol=3,
        )

        # AX2 - Stats
        ymax = max(max(self.quality_overall), max(self.quality_by_iteration))

        if self.optimum is not None:
            ymin = self.optimum.weight
        else:
            ymin = min(min(self.quality_overall), min(self.quality_by_iteration))

        margin = (ymax - ymin) * 0.5
        ax2.set(
            xlim=(-0.5, self.max_iterations + 0.5), ylim=(ymin - margin, ymax + margin)
        )

        iterations = list(range(0, self.iteration + 1))
        ax2.plot(iterations, self.quality_by_iteration, marker="", color="red")
        ax2.plot(iterations, self.quality_overall, marker="", color="grey")

        if self.optimum is not None:
            ax2.axhline(y=self.optimum.weight, color="yellowgreen", linewidth=2)

        # Legend
        red_dot = mlines.Line2D([], [], color="red", marker="", label="Iteration best")
        grey_dot = mlines.Line2D([], [], color="grey", marker="", label="Overall best")
        ax2_handles = [red_dot, grey_dot]

        if self.optimum is not None:
            baseline = mlines.Line2D(
                [], [], color="yellowgreen", linewidth=2, label="Known Optimum"
            )
            ax2_handles.append(baseline)

        ax2.legend(
            handles=ax2_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            shadow=True,
            ncol=2,
        )

        fig.tight_layout()

        # Saving to specific directory and file

        out_path = "output/{}".format(self.output_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)

        plt.savefig("{}/best.png".format(out_path), format="png")
        plt.savefig("{}/best.pgf".format(out_path), format="pgf")
        plt.close(fig)

    def print_map_only(self):

        # make figure, axes
        plt.style.use("ggplot")
        plt.tight_layout()

        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, (ax1) = plt.subplots(
            figsize=(4.4, 5.9), ncols=1, nrows=1, gridspec_kw=gs_kw
        )

        titel = f"eil51 - Iteration {self.iteration}"
        ax1.set(title=titel, xlabel="x", ylabel="y")
        ax1.set_aspect("equal")

        # AX1 Best Solution vs. Optimal Solution

        # Nodes
        xs, ys = zip(*self.problem.node_coords.values())
        # labels = self.problem.node_coords.keys()
        ax1.scatter(xs, ys, marker="o", color="dimgrey", zorder=10)
        # for label, x, y in zip(labels, xs, ys):
        #     ax1.annotate(label, xy=(x, y), zorder=20)
        # xs,ys hold data for city coordinates
        xs, ys = zip(*self.problem.node_coords.values())
        labels = self.problem.node_coords.keys()

        # plots best tour in self.memory
        best_tour = min(self.memory, key=lambda p: p.weight).tour
        xt = []
        yt = []
        for p in best_tour:
            coords = self.problem.node_coords[p]
            xt.append(coords[0])
            yt.append(coords[1])
        xt.append(xt[0])
        yt.append(yt[0])
        ax1.plot(xt, yt, alpha=1.0, color="C1", linestyle="dashed", zorder=2)

        # plots optimum tour if given
        if self.optimum is not None:
            opt_tour = self.optimum.tour
            xt = []
            yt = []
            for p in opt_tour:
                coords = self.problem.node_coords[p]
                xt.append(coords[0])
                yt.append(coords[1])
            xt.append(xt[0])
            yt.append(yt[0])
            ax1.plot(xt, yt, alpha=0.4, color="C4", linewidth=5, zorder=1)

        # Labels
        redline = mlines.Line2D(
            [], [], color="C1", linestyle="dashed", label="Overall Best Tour"
        )
        yellowline = mlines.Line2D(
            [], [], color="C4", linewidth="5", label="Known Optimal Tour"
        )
        grey_dot = mlines.Line2D(
            [], [], color="dimgrey", marker="o", linestyle="", label="City"
        )
        ax1.legend(
            handles=[redline, yellowline, grey_dot],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            shadow=True,
            ncol=1,
        )

        fig.tight_layout(rect=[0, 0, 1, 1])

        # Saving to specific directory and file
        out_path = "output/{}".format(self.output_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        plt.savefig("{}/best_{}.png".format(out_path, self.iteration), format="png")
        plt.savefig("{}/best_{}.pgf".format(out_path, self.iteration), format="pgf")
        plt.close(fig)

    def print_stats_only(self):

        # make figure, axes
        plt.style.use("ggplot")
        plt.tight_layout()

        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, (ax2) = plt.subplots(figsize=(9, 4.5), ncols=1, nrows=1, gridspec_kw=gs_kw)

        ax2.set(title="Quality development", xlabel="iteration", ylabel="tour length")

        ymax = max(max(self.quality_overall), max(self.quality_by_iteration))

        if self.optimum is not None:
            ymin = self.optimum.weight
        else:
            ymin = min(min(self.quality_overall), min(self.quality_by_iteration))

        margin = (ymax - ymin) * 0.5
        ax2.set(
            xlim=(-0.5, self.max_iterations + 0.5), ylim=(ymin - margin, ymax + margin)
        )

        iterations = list(range(0, self.iteration + 1))
        ax2.plot(iterations, self.quality_by_iteration, marker="", color="red")
        ax2.plot(iterations, self.quality_overall, marker="", color="grey")

        if self.optimum is not None:
            ax2.axhline(y=self.optimum.weight, color="yellowgreen", linewidth=2)

        # Legend
        red_dot = mlines.Line2D([], [], color="red", marker="", label="Iteration best")
        grey_dot = mlines.Line2D([], [], color="grey", marker="", label="Overall best")
        ax2_handles = [red_dot, grey_dot]

        if self.optimum is not None:
            baseline = mlines.Line2D(
                [], [], color="yellowgreen", linewidth=2, label="Known Optimum"
            )
            ax2_handles.append(baseline)

        ax2.legend(handles=ax2_handles, loc="upper right", shadow=True)

        # Saving to specific directory and file
        out_path = "output/{}".format(self.output_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        plt.savefig("{}/stats.png".format(out_path), format="png")
        plt.savefig("{}/stats.pgf".format(out_path), format="pgf")
        plt.close(fig)

    def print_state(self, population):
        """
        Print State of Optimization with Coordinate System of tours and stats,
        default: only latest addition to the memory is plottet
        a population given by list of tours is additionally plottet if provided
        """

        # make figure, axes
        # plt.style.use('seaborn-whitegrid')
        plt.style.use("ggplot")

        gs_kw = dict(width_ratios=[3, 2], height_ratios=[1])
        fig, (ax1, ax2) = plt.subplots(
            figsize=(9, 4.5), ncols=2, nrows=1, gridspec_kw=gs_kw
        )
        plt.tight_layout()

        ax1.set(title="Optimisation State", xlabel="x", ylabel="y")
        ax1.set_aspect("equal")
        ax2.set(title="Quality development", xlabel="iteration", ylabel="tour length")

        # AX1 - Coordinate System

        # Nodes
        xs, ys = zip(*self.problem.node_coords.values())
        labels = self.problem.node_coords.keys()
        ax1.scatter(xs, ys, marker="o", color="dimgrey", zorder=10)
        for label, x, y in zip(labels, xs, ys):
            ax1.annotate(label, xy=(x, y), zorder=20)

        # Tours (in current population)
        for _ in range(0, len(population)):
            for agent in population:
                xt = []
                yt = []
                for p in agent.tour:
                    coords = self.problem.node_coords[p]
                    xt.append(coords[0])
                    yt.append(coords[1])
                xt.append(xt[0])
                yt.append(yt[0])
                ax1.plot(xt, yt, alpha=0.1, color="goldenrod", linewidth=5)

        # Best Tour in Population
        best_agent = self.memory[self.iteration]
        best_tour = best_agent.tour
        xt = []
        yt = []
        for p in best_tour:
            coords = self.problem.node_coords[p]
            xt.append(coords[0])
            yt.append(coords[1])
        xt.append(xt[0])
        yt.append(yt[0])
        ax1.plot(xt, yt, alpha=1.0, color="darkred", linestyle="dashed")

        # LABELS
        redline = mlines.Line2D(
            [], [], color="darkred", linestyle="dashed", label="Best Tour in Iteration"
        )
        yellowline = mlines.Line2D(
            [], [], color="goldenrod", linewidth="5", label="Other Tours in Iteration"
        )
        grey_dot = mlines.Line2D(
            [], [], color="dimgrey", marker="o", linestyle="", label="Node"
        )

        ax1.legend(
            handles=[grey_dot, yellowline, redline],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            shadow=True,
            ncol=3,
        )

        # AX2 - Stats
        ax2.set(xlim=(0 - 0.5, self.max_iterations + 0.5))
        iterations = list(range(0, self.iteration + 1))

        ax2.plot(
            iterations, self.quality_by_iteration, marker="o", color="red", linestyle=""
        )
        ax2.plot(
            iterations, self.quality_overall, marker="x", color="grey", linestyle=""
        )

        # LABELS
        red_dot = mlines.Line2D(
            [], [], color="red", marker="o", label="Iteration best", linestyle=""
        )
        grey_dot = mlines.Line2D(
            [], [], color="grey", marker="x", label="Overall best", linestyle=""
        )
        ax2.legend(
            handles=[red_dot, grey_dot],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            shadow=True,
            ncol=2,
        )

        fig.tight_layout()

        # Saving to specific directory and file

        out_path = "output/{}".format(self.output_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            "{}/iteration_{:03d}.png".format(out_path, self.iteration), format="png"
        )
        plt.savefig(
            "{}/iteration_{:03d}.pgf".format(out_path, self.iteration), format="pgf"
        )
        plt.close(fig)
