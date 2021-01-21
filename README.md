# bachelorthesis-bioinspired-optimisation

## Use

Code works with .tsp files with EDGE_WEIGHT_TYPE: EUC_2D (Weights are Euclidean distances in 2-D) and (immplicit) NODECOORDTYPE: TWOD_COORDS (Nodes are specified by coordinates in 2-D) for which .opt.tour file is provided.

(a number of instances from the TSPLIB http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ are included)

You may adjust:
- algorithm parameters in the *_tests.py files
- output settings in the run() function in the *_optimisation.py files

```python abc_tests.py```

## Output Settings

Both .png and .pgf files will be created.

Tour Coordinate System view may produce mixed results depending on tsp instance structure and size.

```print_states=True```

At each iteration, the tour coordinate system will be printed.

```print_joint_result=True```

The final results will be printed in joint view of tour coordinate system and quality development graph.

```print_result_map=True```

The final results will be printed in tour coordinate system view.
```print_result_stats=True```

The final results will be printed in quality development graph view.