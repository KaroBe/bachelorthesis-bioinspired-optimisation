"""
FA testing
"""

# ----------------------------------------------------------

from testseries import TestSeries

# ----------------------------------------------------------


def test():
    filename = "instnaces/eil51.tsp"

    fa_parameters = {
        "algorithm": "FA",
        "heuristic_init": True,
        "size": 5,
        "max_iterations": 5,
        "problem_filename": filename,
        "m": 11,
        "gamma": 0.007,
    }

    # Run test series of a number of test runs with specified parameters
    experiment_fa = TestSeries()
    experiment_fa.run_test_series(fa_parameters, runs=1)


# ----------------------------------------------------------

if __name__ == "__main__":
    test()
