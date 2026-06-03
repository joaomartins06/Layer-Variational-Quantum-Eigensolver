"""
Import a cloud-generated results JSON into the local MLflow database.

Usage:
    python -m experiments.maxcut.log_results results_lvqe_2layers.json
    python -m experiments.maxcut.log_results results_basevqe_1layers.json
"""

import sys
import json
import numpy as np
import mlflow

from src.logging_utils import start_run, nested_run, log_figure, log_metrics_series
from experiments.maxcut.finite_sampling import make_ratio_plot, make_loss_plot


def replay(json_path: str) -> None:
    with open(json_path) as f:
        data = json.load(f)

    experiment   = data["experiment"]
    params       = data["params"]
    instance_seeds = data["instance_seeds"]
    run_seeds    = data["run_seeds"]
    results_list = data["results"]

    results = {(r["instance_seed"], r["run_seed"]): r for r in results_list}

    all_ratios = [results[(iseed, rseed)]["seed_ratios"]
                  for iseed in instance_seeds for rseed in run_seeds]
    all_losses = [results[(iseed, rseed)]["seed_losses"]
                  for iseed in instance_seeds for rseed in run_seeds]
    instance_best_ratios = [
        max(results[(iseed, rseed)]["final_approx_ratio"] for rseed in run_seeds)
        for iseed in instance_seeds
    ]

    with start_run(experiment, params):

        for iseed in instance_seeds:
            with nested_run(f"instance_{iseed}", {"instance_seed": iseed}):
                for rseed in run_seeds:
                    r = results[(iseed, rseed)]
                    with nested_run(f"run_{rseed}", {"instance_seed": iseed, "run_seed": rseed}):
                        for layer, ratio in enumerate(r["seed_ratios"], start=1):
                            mlflow.log_metric("approx_ratio", ratio, step=layer)
                        log_metrics_series("optimizer_loss", np.concatenate(r["seed_losses"]))
                        mlflow.log_metric("final_approx_ratio", r["final_approx_ratio"])

        all_ratios_arr = np.array(all_ratios)
        finals = np.array(instance_best_ratios)
        n = len(finals)

        mlflow.log_metrics({
            "mean_final_approx_ratio": float(finals.mean()),
            "sem_final_approx_ratio":  float(finals.std(ddof=1) / np.sqrt(n)),
            "max_final_approx_ratio":  float(finals.max()),
            "min_final_approx_ratio":  float(finals.min()),
        })

        n_runs    = params["n_runs"]
        n_nodes   = params["num_nodes"]
        n_layers  = params["n_layers"]
        k_per_layer = params["k_per_layer"]

        log_figure(
            make_ratio_plot(all_ratios_arr, instance_seeds, n_runs, n_nodes),
            "approx_ratio_vs_layers.png"
        )
        log_figure(
            make_loss_plot(all_losses, instance_seeds, n_runs, n_layers, k_per_layer),
            "optimizer_loss_vs_iterations.png"
        )

    print(f"Logged to MLflow experiment '{experiment}'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    replay(sys.argv[1])
