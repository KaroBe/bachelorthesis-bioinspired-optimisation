"""
Test Series Class allowing automated execution of test series consisting of a number of runs
"""

from fa_optimisation import FA_Optimisation
from abc_optimisation import ABC_Optimisation
from aco_optimisation import ACO_Optimisation

import csv

import cProfile
import pstats
from io import StringIO


class TestSeries:
    """
    Tool for executing Test Series of Optimisation.
    """

    # Initialization and built-in function overriding

    def __init__(self):
        self.data = []  # results of the profiler
        self.quality_development_overall = []
        self.quality_development_iteration = []

    def run(self, parameters, output_path, runs=1):
        """
        Supporting function that performs a number of runs of specified optimisation.

        Data recorded in TestSeries 
            quality developement
            profiler output (stats of function calls and time)
        """
        for run in range(0, runs):
            print(f'Run {run:<3} of {runs}{"":<20}')
            run_path = f"{output_path}/run_{run:03d}"

            opt = None
            profiler = cProfile.Profile()

            if parameters["algorithm"] == "FA":
                profiler.enable()
                opt = FA_Optimisation(parameters, run_path)

            elif parameters["algorithm"] == "ABC":
                profiler.enable()
                opt = ABC_Optimisation(parameters, run_path)

            elif parameters["algorithm"] == "ACO":
                profiler.enable()
                opt = ACO_Optimisation(parameters, run_path)

            stats, quality_iteration, quality_overall = opt.run()
            profiler.disable()

            # append results of optimisation run to testseries data
            self.data.append(stats)
            self.quality_development_overall.append(quality_iteration)
            self.quality_development_iteration.append(quality_overall)

            # output profiler stats for run to csv

            # include profiler data for only the functions contained in the following regex:
            script_regex = "abc_optimisation\.py|fa_optimisation\.py|aco_optimisation\.py|ant\.py|firefly\.py|foodsource\.py|optimisation_base\.py|candidate_solution\.py"

            result_stream = StringIO()
            profiler_stats = pstats.Stats(profiler, stream=result_stream)
            profiler_stats.strip_dirs()
            profiler_stats.sort_stats("cumulative").print_stats(script_regex)
            result = result_stream.getvalue()
            # chop the string into a csv-like buffer
            result = "ncalls" + result.split("ncalls")[-1]
            result = "\n".join(
                [",".join(line.rstrip().split(None, 6))
                 for line in result.split("\n")]
            )
            # save it to disk
            with open(f"output/{run_path}/profile.csv", "w") as f:
                f.write(result)
                print("writing the result")

        self.save_to_csv(output_path)

    def run_test_series(self, parameters, output_path=None, runs=1):
        """
        Runs Test Series where specified optimisation is performed a number of times
        and the results are accumulated and output to a csv file.
        Takes parameters and optional output path and number of runs.
        Custom Output Path may be specified, otherwise will be generated from
        parameters. Per default, one run will be performed.
        """
        if output_path is None:
            output_path = path_from_parameters(parameters)
        self.run(parameters, output_path, runs=runs)

    # def run_parameter_test(self, parameters, variable, range, step, output_path=None, runs=1):
    #     """
    #     Runs Test Series where specified optimisation is performed a number of times
    #     for each value in the specified range for a parameter.
    #     and the results are accumulated and output to a csv file.
    #     Takes parameters and number of runs.
    #     Custom Output Path may be specified, otherwise will be generated from
    #     parameters.
    #     """
    #     if output_path is None:
    #         output_path = path_from_parameters(parameters)
    #     # TODO: Adapt for any parameter
    #     value = range[0]
    #     run = 0
    #     while value < range[1]:
    #         print(f"Parameter {variable} = {value:<6} of {range[1]}")
    #         run_path = f"{output_path}/{variable}_{value:03d}_"
    #         # TODO: run with parameter spcified in 'variable' changed
    #         # gamma = value etc.
    #         self.run(parameters, output_path=run_path)
    #         value += step
    #         run += 1
    #     self.save_to_csv(output_path)

    def save_to_csv(self, output_path):
        """
        Saves data accumulated in TestSeries Object to csv file.

        Output:
            results.csv
            quality_development.csv
        """
        with open(f"output/{output_path}/results.csv", "w", newline="") as csvfile:

            entries = self.data
            fieldnames = list(self.data[0].keys())

            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writeheader()

            for entry in entries:
                thewriter.writerow(entry)

        with open(
            f"output/{output_path}/quality_development.csv", "w", newline=""
        ) as csvfile:
            max_len = len(
                max(self.quality_development_overall, key=lambda l: len(l)))
            entries = []
            for iteration in range(0, max_len):
                temp = {"iteration": iteration}
                for run in range(0, len(self.quality_development_overall)):
                    quality = None
                    if iteration <= len(self.quality_development_overall[run]):
                        quality = self.quality_development_overall[run][iteration]
                    temp[f"quality_{run:02d}"] = quality
                entries.append(temp)

            fieldnames = entries[0].keys()

            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writeheader()

            for entry in entries:
                thewriter.writerow(entry)

        with open(
            f"output/{output_path}/iteration_development.csv", "w", newline=""
        ) as csvfile:
            max_len = len(
                max(self.quality_development_iteration, key=lambda l: len(l)))
            entries = []
            for iteration in range(0, max_len):
                temp = {"iteration": iteration}
                for run in range(0, len(self.quality_development_iteration)):
                    quality = None
                    if iteration <= len(self.quality_development_iteration[run]):
                        quality = self.quality_development_iteration[run][iteration]
                    temp[f"quality_{run:02d}"] = quality
                entries.append(temp)

            fieldnames = entries[0].keys()

            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writeheader()

            for entry in entries:
                thewriter.writerow(entry)

        print(f"Output saved to output/{output_path}")


def path_from_parameters(parameters):
    """
    Returns path generated from parameters for ease of use.
    """
    filename = parameters["problem_filename"]
    size = parameters["size"]
    max_iterations = parameters["max_iterations"]

    if parameters["algorithm"] == "FA":
        gamma = parameters["gamma"] * 100
        m = parameters["m"]
        return f"FA_{filename[:-4]}_s{size}_i{max_iterations}"

    elif parameters["algorithm"] == "ABC":
        # TODO: Better out_path with all parameters
        return f"ABC_{filename[:-4]}_s{size}_i{max_iterations}"

    elif parameters["algorithm"] == "ACO":
        # TODO: Better out_path with all parameters
        return f"ACO_{filename[:-4]}_s{size}_i{max_iterations}"
