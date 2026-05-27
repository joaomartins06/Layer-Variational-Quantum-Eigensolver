"""
Generic MLflow logging utilities.

Quick-start
-----------
    from src.logging_utils import start_run, log_figure

    with start_run("my-experiment", {"lr": 0.01, "layers": 3}):
        # ... training loop ...
        mlflow.log_metric("loss", loss, step=i)
        log_figure(fig, "loss_curve.png")

View all runs:
    mlflow ui   →  http://localhost:5000

Shared tracking (e.g. DagsHub):
    export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt
import mlflow


@contextmanager
def start_run(experiment_name: str, params: dict[str, Any] | None = None):
    """
    Context manager that sets the experiment, starts a run, logs params,
    and ensures the run is ended cleanly — even if an exception is raised.

    Parameters
    ----------
    experiment_name : MLflow experiment to log into (created if absent).
    params          : Flat dict of hyper-parameters / config to record.

    Example
    -------
    with start_run("lvqe-maxcut", {"n_layers": 2, "optimizer": "SMO"}):
        mlflow.log_metric("loss", 0.42, step=1)
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        yield


@contextmanager
def nested_run(run_name: str, params: dict[str, Any] | None = None):
    """
    Context manager for a nested child run inside an already-active parent run.

    Parameters
    ----------
    run_name : Display name for this child run (e.g. "seed_42").
    params   : Optional params specific to this child run.

    Example
    -------
    with start_run("my-exp", global_params):
        for seed in seeds:
            with nested_run(f"seed_{seed}", {"seed": seed}):
                mlflow.log_metric("loss", ..., step=...)
    """
    with mlflow.start_run(run_name=run_name, nested=True):
        if params:
            mlflow.log_params(params)
        yield


def log_figure(fig: plt.Figure, artifact_path: str) -> None:
    """
    Save a matplotlib Figure as an MLflow artifact and close it.

    Parameters
    ----------
    fig           : The Figure to save.
    artifact_path : Filename inside the run's artifact store, e.g. "loss.png".
    """
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def log_metrics_series(name: str, values, *, start_step: int = 0) -> None:
    """
    Log an iterable of scalar values as successive steps of one metric.

    Parameters
    ----------
    name       : Metric name in MLflow.
    values     : Any iterable of floats (list, np.ndarray, generator, …).
    start_step : Step index to begin counting from (default 0).

    Example
    -------
    log_metrics_series("optimizer_loss", loss_array)
    log_metrics_series("optimizer_loss", loss_array, start_step=100)
    """
    for i, v in enumerate(values, start=start_step):
        mlflow.log_metric(name, float(v), step=i)