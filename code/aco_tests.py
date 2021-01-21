"""
ACO testing
"""

# ----------------------------------------------------------

from testseries import TestSeries

# ----------------------------------------------------------


def test():
    filename = "instances/test10.tsp"

    aco_parameters = {
        "algorithm": "ACO",
        "heuristic_init": True,
        "size": 5,
        "max_iterations": 5,
        "problem_filename": filename,
        "alpha": 1,
        "beta": 2,
        "tau_zero": 1,
        "phi": 0.1,
        "rho": 0.1,
        "q_zero": 0.9,
    }

    # Run test series of a number of test runs with specified parameters
    experiment_aco = TestSeries()
    experiment_aco.run_test_series(aco_parameters, runs=1)


# ----------------------------------------------------------

if __name__ == "__main__":
    test()
