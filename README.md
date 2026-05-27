# Layer-Variational-Quantum-Eigensolver
Analysis of the Layer Variational Quantum Eigensolver applied to graph problems and its extension to new platforms.

## Experiment tracking

This project uses MLflow for local experiment tracking. Each run of an experiment from `experiments/<...>/` automatically logs hyperparameters, performance metrics, plots, etc.

### Setup
From `experiments/<...>/`, run `<experiment>.py` with:
```bash
python <experiment>.py
```
Results are then saved to `experiments/<...>/mlruns/`. They can be visualized by running:
```bash
mlflow ui
```
and opening `http://localhost:5000` in a web browser.

Experiments which average results over multiple runs are logged as one top-level run with child runs for each different seed. This allows for individual inspection or comparison across seeds.

> [!IMPORTANT]  
> `mlruns/` and `mlflow.db` are gitignored: runs stay *local* to your machine.