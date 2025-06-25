# Contrail Optimization and Analysis

This repository contains the code processing pipeline for the analyses presented in the paper associated with the paper: **Contrail, or not contrail, that is the question: the “feasibility” of climate-optimal routing**.

## Repository Structure

- `01_pre_process.py` — Preprocesses raw flight data, filters trajectories, merges meteorological data, and computes contrail-relevant conditions.
- `02_gen_cost_grids.py` — Generates and smooths cost grids (ERA5 and ARPEGE) for contrail persistence and related meteorological parameters.
- `03_optimization.py` — Runs trajectory optimization for fuel and contrail minimization using the generated cost grids.
- `04_post_optimization.py` — Post-processes optimized trajectories and recomputes contrail conditions.
- `05_plots.py` — Contains plotting scripts for visualizing cost grids, trajectories, and optimization results as shown in the paper.
- `06_occupancy.py` — Analyzes airspace occupancy for different trajectory scenarios and generates related figures.
- `data/` — download the data files from figshare (<https://doi.org/10.6084/m9.figshare.29400650>) and store them in this directory. Create the `data` directory if it does not exist.

## Requirements

- Python 3.10+
- Main dependencies: 
    - `openap` for OpenAP aircraft performance model
    - `openap-top` for trajectory optimization
    - `fastmeteo` for meteorological data processing
    - `traffic` for trajectory optimization and analysis


Install dependencies (example using pip):

```bash
pip install openap openap-top fastmeteo traffic
```

## Notes
- Some scripts are computationally intensive and may require significant memory and CPU resources.
- Data files  are not included and must be obtained separately from figshare: <https://doi.org/10.6084/m9.figshare.29400650>

