"""
ABC testing
"""

# ----------------------------------------------------------

from testseries import TestSeries

# ----------------------------------------------------------


def test():
    filename = "instances/test10.tsp"

    abc_parameters = {
        "algorithm": "ABC",
        "heuristic_init": True,
        "size": 5,
        "max_iterations": 5,
        "problem_filename": filename,
        "reconnection_probabilty": 0.5,
        "correction_perturbation_probability": 0.8,
        "linearity_probablity": 0.2,
        "max_neighbourhood": 5,
        "L": 2,
    }
    # Fixed Parameters: Lmin=2; Lmax = n/2 instead of sqrt(n) (Compare GSTM by Albayrak and
    # Allhaverdi 2011 vs. usage of GSTM operator in qCBCO by Karaboga and Gorkemli 2013)

    # Run test series of a number of test runs with specified parameters
    experiment_abc = TestSeries()
    experiment_abc.run_test_series(abc_parameters, runs=1)


# ----------------------------------------------------------

if __name__ == "__main__":
    test()
