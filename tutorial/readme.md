# Tutorials

This directory contains a set of Jupyter notebooks that serve as tutorials for the ATRD Symposium. The notebooks are designed to help users understand how to set up and run trajectory optimizations, visualize results, and work with large datasets using multi-processing.

0. `0_warm_up.ipynb` - Notebook provides a brief introduction to trajectory optimization using TOP, getting wind data, and using 4D cost grids.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junzis/contrail-or-not/blob/main/tutorial/0_warm_up.ipynb)

1. `1_data_walk_through.ipynb` - Notebook walks through the data used or generated in the study, including trajectories, cost grids from ERA5 and ARPEGE.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junzis/contrail-or-not/blob/main/tutorial/1_data_walk_through.ipynb)

2. `2_single_optimal_trajectory.ipynb` - Notebook demonstrates how to optimize a single trajectory using TOP, including setting up the optimization problem, running the optimization, and visualizing the results.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junzis/contrail-or-not/blob/main/tutorial/2_single_optimal_trajectory.ipynb)

3. `3_multi_processing.ipynb` - Notebook shows how to run multiple trajectory optimizations in parallel using traffic's multiprocessing procedure, which is useful for large-scale studies.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junzis/contrail-or-not/blob/main/tutorial/3_multi_processing.ipynb)