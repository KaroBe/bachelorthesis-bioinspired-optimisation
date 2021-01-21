# Bio-Inspired Algorithms for Combinatorial Optimisation Problems
## Metaheuristic Characteristics in Ant Colony Optimisation, Artificial Bee Colony and Firefly Algorithm 

In this thesis presented for the Bachelors degree, three bio-inspired metaheuristics for the travelling salesperson problem were implemented and tested:

Artificial Bee Colony (ABC) - Combinatorial Artificial Bee Colony [1,2]

Ant Colony Optimisation (ACO) - Ant Colony System [3]

Firefly Algorithm (FA) - Discrete Firefly Algorithm [4]

The code is licensed unter [MIT license](bachelorthesis-bioinspired-optimisation/LICENSE).

## Use

Implememtation in ***Python 3***.

Code works with .tsp files with ```EDGE_WEIGHT_TYPE: EUC_2D``` (Weights are Euclidean distances in 2-D) and (immplicit) ```NODECOORDTYPE: TWOD_COORDS``` (Nodes are specified by coordinates in 2-D) for which .opt.tour file is provided.

(a number of instances from the TSPLIB http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ are included, as well as small "test10" instance)

The [tsplib95 library](https://github.com/rhgrant10/tsplib95) for python is used for handling .tsp and .opt.tour files (```pip install tsplib95```).

You may adjust:
- algorithm parameters in the *_tests.py files
- output settings in the run() function in the *_optimisation.py files

Run ```python *_tests.py```.

## Output Settings

Both .png and .pgf files will be created.

Tour Coordinate System view may produce mixed results depending on tsp instance structure and size.

```print_states=True``` - At each iteration, the tour coordinate system will be printed.

```print_joint_result=True``` - The final results will be printed in joint view of tour coordinate system and quality development graph.

```print_result_map=True``` - The final results will be printed in tour coordinate system view.

```print_result_stats=True``` - The final results will be printed in quality development graph view.

## References

All sources used for the thesis are provided [as .bib file](bachelorthesis-bioinspired-optimisation/bibliography.bib)

### Artificial Bee Colony
[1] D. Karaboga and B. Gorkemli, “A combinatorial Artificial Bee Colony algorithm for traveling salesman problem,” 2011 International Symposium on Innovations in Intelligent Systems and Applications, pp. 50–53, 2011.

[2] D. Karaboga and B. Gorkemli, “Solving Traveling Salesman Problem by Using Combinatorial Artificial Bee Colony Algorithms,” International Journal on Artificial Intelligence Tools, vol. 28, no. 01, Art. no. 01, 2019, doi: 10.1142/S0218213019500040.

### Ant Colony Optimisation

[3] M. Dorigo and L. M. Gambardella, “Ant colony system: a cooperative learning approach to the traveling salesman problem,” IEEE Transactions on Evolutionary Computation, vol. 1, no. 1, Art. no. 1, 1997-04, doi: 10.1109/4235.585892.

### Firefly Algorithm

[4] G. K. Jati, R. Manurung, and Suyanto, “Discrete Firefly Algorithm for Traveling Salesman Problem: A New Movement Scheme,” in Swarm Intelligence and Bio-Inspired Computation, Ed.X.-S. Yang, Z. Cui, R. Xiao, A. H. Gandomi, and M. Karamanoglu Oxford: Elsevier, 2013, pp. 295–312. doi: 10.1016/B978-0-12-405163-8.00013-2
