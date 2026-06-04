### Applied Quantum Algorithms Projects
***
 
# Layer VQE

Student project for the **Applied Quantum Algorithms** course (Delft, Q3-Q4 2026).

---

## Overview

Implementation and experimental evaluation of the **Layer VQE (L-VQE)** algorithm from [Liu et al. (2022)](https://ieeexplore.ieee.org/document/9669165), applied to the k-Community Detection and the Max-Cut problems.

L-VQE grows the ansatz one layer at a time, warm-starting each new layer from the previously optimized parameters. According to the paper, this staged construction avoids the barren-plateau and local-minima traps that afflict randomly initalized deep circuits, and is more robust to finite-sampling noise than standard fixed-ansatz VQE.

---

## Repository structure

```
src/
  problem.py                # Abstract base class for combinatorial optimization problems
  community_detection.py    # k-Community Detection problem
  maxcut.py                 # Max-Cut problem
  ansatze.py                # Layered ansatz
  lvqe.py                   # L-VQE algorithm
  basevqe.py                # Fixed-ansatz VQE (baseline)
  simulator.py              # Tensor-network simulator via quimb (exact or finite-sampling)
  optimizer.py              # COBYLA, SMO, and Adam optimizers
  logging_utils.py          # MLflow helpers

experiments/maxcut/
  finite_sampling.py        # L-VQE experiment (finite sampling, parallelized)
  base_finite_sampling.py   # Fixed-ansatz VQE experiment (finite sampling, parallelized)
  log_results.py            # Replay a results JSON into the local MLflow database
  layers.py                 # Ablation over number of layers
  schwagerl.py              # Benchmark, following Schwagerl et al. (2026) approach
  qaoa_comparison.py        # QAOA comparison
  plot_model_comparison.py  # Cross-model result plots

notebooks/              # Exploratory notebooks
quantum_walk/           # Quantum-walk baseline
results/
liu.pdf                 # Reference paper
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**IBM Quantum (optional, for hardware runs):**

```bash
cp .env.example .env
# fill in IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE
```

---

## Running experiments

All commands should be run from the **project root**.

### Locally

```bash
# L-VQE with finite sampling
VQE_N_LAYERS=2 python -m experiments.maxcut.finite_sampling

# Base-VQE baseline
VQE_N_LAYERS=1 python -m experiments.maxcut.base_finite_sampling
```

`VQE_N_LAYERS` controls the number of layers added on top of the initial layer-0 ansatz.

Results are logged directly into MLflow as they run. See [Experiment tracking](#experiment-tracking) below.

### Via GitHub Actions (remote)

The workflow `vqe_experiment.yml` is triggered manually from the **Actions** tab:

1. Go to **Actions → VQE Experiment → Run workflow**
2. Select the script (`finite_sampling` or `base_finite_sampling`) and the number of layers
3. Once the run finishes, download the JSON artifact from the workflow summary page (`results-<script>-<N>layers`)
4. Replay it into your local MLflow database:

```bash
python -m experiments.maxcut.log_results results_lvqe_2layers.json
# or
python -m experiments.maxcut.log_results results_basevqe_1layers.json
```

> The JSON artifact contains all raw results (per-seed ratios and loss histories). `log_results.py` reconstructs the full nested MLflow run — metrics, parameters, and plots — exactly as if the experiment had run locally.

---

## Experiment tracking

This project uses [MLflow](https://mlflow.org) for local experiment tracking. Each run logs hyperparameters, per-layer approximation ratios, optimizer loss trajectories, and summary plots.

Runs are structured as nested MLflow runs:

```
parent run  (experiment-level params + aggregate metrics + plots)
└── instance_<seed>
    └── run_<seed>   (per-seed approx_ratio, optimizer_loss, final_approx_ratio)
```

### Viewing results

```bash
mlflow ui   # then open http://localhost:5000
```

Run this from the **project root** to see all tracked experiments.

> [!IMPORTANT]
> `mlruns/` and `mlflow.db` are gitignored — runs stay local to your machine.

---

## Algorithm overview

# TODO
