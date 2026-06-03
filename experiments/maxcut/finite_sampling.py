import os
import json
import contextlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mlflow
from concurrent.futures import ProcessPoolExecutor

from src.maxcut import MaxCut
from src.simulator import QuimbSimulator
from src.optimizer import COBYLA, SMO
from src.lvqe import LayerVQE
from src.logging_utils import start_run, nested_run, log_figure, log_metrics_series



# ── Config ─────────────────────────────────────────────────────────────────────
N_NODES = 52
N_INSTANCES = 10
N_RUNS = 3

N_LAYERS = int(os.environ.get("VQE_N_LAYERS", 2))
K_PER_LAYER = 50
K_FINAL = 400

SIMULATOR = SMO

USE_SAMPLING = True
N_SAMPLES = 50

PARAMS = dict(
    num_nodes = N_NODES,
    num_instances = N_INSTANCES,
    n_runs = N_RUNS,
    n_layers = N_LAYERS,
    k_per_layer = K_PER_LAYER,
    k_final = K_FINAL,
    optimizer = SIMULATOR.__name__,
    use_sampling = USE_SAMPLING,
    n_samples = N_SAMPLES,
)

rng = np.random.default_rng(42)
instance_seeds = rng.integers(0, 10000, size=N_INSTANCES).tolist()
run_seeds = rng.integers(0, 10000, size=N_RUNS).tolist()


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_random_graph(N: int, seed: int) -> nx.Graph:
    assert N % 2 == 0
    rng_graph = np.random.RandomState(seed)
    while True:
        G = nx.random_regular_graph(3, N, seed=rng_graph)
        if nx.is_connected(G):
            return G


@contextlib.contextmanager
def _silence():
    """Suppress stdout/stderr in worker processes (hides interleaved tqdm bars)."""
    with open(os.devnull, 'w') as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        yield


def _single_run(args):
    """
    Run one (graph, instance_seed, run_seed, best_known_value) job.
    No MLflow calls — safe to run in a worker process.
    best_known_value is pre-computed in the main process to avoid calling
    qiskit_optimization (GW solver) inside workers.
    """
    graph, instance_seed, run_seed, best_known_value = args
    np.random.seed(run_seed)

    problem = MaxCut(graph, seed=instance_seed)
    problem.__dict__["best_known_value"] = best_known_value

    with _silence():
        result = LayerVQE(
            problem = problem,
            simulator = QuimbSimulator(),
            optimizer_class = SIMULATOR,
            seed = run_seed,
            n_layers = N_LAYERS,
            k_per_layer = K_PER_LAYER,
            k_final = K_FINAL,
            use_sampling = USE_SAMPLING,
            n_samples = N_SAMPLES,
            record_loss = True,
        ).run()

    seed_ratios = [result["history"]["approx_ratio"][l]
                   for l in result["history"]["layer"]]
    seed_losses = result["history"]["optimizer_loss"]

    return {
        "instance_seed": instance_seed,
        "run_seed":      run_seed,
        "seed_ratios":   seed_ratios,
        "seed_losses":   seed_losses,
        "final_approx_ratio": result["final_approx_ratio"],
    }


def make_ratio_plot(
    all_ratios: np.ndarray,
    instance_seeds: list,
    n_runs: int,
    num_nodes: int,
) -> plt.Figure:
    n_instances = len(instance_seeds)
    colors = plt.cm.tab10(np.linspace(0, 1, n_instances))
    n_layers = all_ratios.shape[1]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, seed in enumerate(instance_seeds):
        instance_ratios = all_ratios[i * n_runs : (i + 1) * n_runs]
        mean = instance_ratios.mean(axis=0)
        std  = instance_ratios.std(axis=0)

        for ratios in instance_ratios:
            ax.plot(layers, ratios, color=colors[i], alpha=0.20, linewidth=1, zorder=1)
        ax.fill_between(layers, mean - std, mean + std,
                        alpha=0.18, color=colors[i], zorder=2)
        ax.plot(layers, mean, "o-", color=colors[i], linewidth=2.5,
                markersize=7, label=f"Instance seed {seed}", zorder=3)

        best_final = instance_ratios[:, -1].max()
        ax.annotate(
            f"{best_final:.3f}",
            xy=(layers[-1], best_final),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=colors[i], va="center",
        )

    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.6, label="Optimal (ratio = 1)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Approximation ratio", fontsize=12)
    ax.set_title(
        f"L-VQE approximation ratio vs. layers\n"
        f"(mean ± std over {n_runs} runs, {n_instances} instances, {num_nodes}-node 3-regular graph)",
        fontsize=12,
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([f"Layer {l}" for l in layers], fontsize=9)
    ax.set_ylim(bottom=0.5, top=1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.minorticks_on()
    fig.tight_layout()
    return fig


def make_loss_plot(
    all_losses: list,
    instance_seeds: list,
    n_runs: int,
    n_layers: int,
    k_per_layer: int,
) -> plt.Figure:
    n_instances = len(instance_seeds)
    colors = plt.cm.tab10(np.linspace(0, 1, n_instances))

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, seed in enumerate(instance_seeds):
        instance_losses = all_losses[i * n_runs : (i + 1) * n_runs]
        trajectories = np.array([np.concatenate(run) for run in instance_losses])

        mean = trajectories.mean(axis=0)
        std  = trajectories.std(axis=0)
        xs   = np.arange(len(mean))

        for traj in trajectories:
            ax.plot(xs, traj, color=colors[i], alpha=0.15, linewidth=0.8, zorder=1)
        ax.fill_between(xs, mean - std, mean + std,
                        alpha=0.15, color=colors[i], zorder=2)
        ax.plot(xs, mean, linewidth=2.5, color=colors[i],
                label=f"Instance seed {seed}", zorder=3)

    y_top = ax.get_ylim()[1]
    for idx in range(1, n_layers + 1):
        x = k_per_layer * idx
        ax.axvline(x=x, color="black", linestyle="--", linewidth=1.2, alpha=0.5,
                   label="Layer boundary" if idx == 1 else "")
        ax.text(x + len(mean) * 0.005, y_top, f"L{idx}",
                fontsize=8, ha="left", va="top", color="black", alpha=0.7)

    ax.set_xlabel("Total optimisation iterations", fontsize=12)
    ax.set_ylabel("Energy (loss)", fontsize=12)
    ax.set_title(
        f"Training loss per instance (mean ± std over {n_runs} runs)\n"
        f"{n_instances} instances, {n_layers} layers",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.minorticks_on()
    fig.tight_layout()
    return fig


# ── Experiment ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Generate all graphs and pre-compute best_known_value in the main process.
    # Workers must not call _gw_optimum() (qiskit_optimization unavailable there).
    graphs = {seed: get_random_graph(N_NODES, seed) for seed in instance_seeds}
    best_known = {seed: MaxCut(graphs[seed], seed=seed).best_known_value
                  for seed in instance_seeds}

    # Build flat job list preserving instance × run ordering
    jobs = [
        (graphs[iseed], iseed, rseed, best_known[iseed])
        for iseed in instance_seeds
        for rseed in run_seeds
    ]

    n_workers = min(len(jobs), os.cpu_count() or 4)
    print(f"Launching {len(jobs)} jobs across {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_flat = list(executor.map(_single_run, jobs))

    # Index results for easy lookup during logging
    results = {(r["instance_seed"], r["run_seed"]): r for r in results_flat}

    # Print summary
    for iseed in instance_seeds:
        for rseed in run_seeds:
            r = results[(iseed, rseed)]
            print(f"  instance={iseed}  run={rseed}  "
                  f"final_approx_ratio={r['final_approx_ratio']:.4f}")

    # ── Save raw results to JSON (for cloud → local MLflow import) ──────────────
    json_path = os.path.join(
        os.path.dirname(__file__),
        f"results_lvqe_{N_LAYERS}layers.json"
    )
    with open(json_path, "w") as f:
        json.dump({
            "experiment": "lvqe-maxcut-schwagerl",
            "params": PARAMS,
            "instance_seeds": instance_seeds,
            "run_seeds": run_seeds,
            "results": [
                {**r, "seed_losses": [arr.tolist() for arr in r["seed_losses"]]}
                for r in results_flat
            ],
        }, f)
    print(f"Results saved → {json_path}")

    # ── MLflow logging (sequential, main process only) ──────────────────────────
    all_ratios = [results[(iseed, rseed)]["seed_ratios"]
                  for iseed in instance_seeds for rseed in run_seeds]
    all_losses = [results[(iseed, rseed)]["seed_losses"]
                  for iseed in instance_seeds for rseed in run_seeds]
    instance_best_ratios = [
        max(results[(iseed, rseed)]["final_approx_ratio"] for rseed in run_seeds)
        for iseed in instance_seeds
    ]

    with start_run("lvqe-maxcut-schwagerl", PARAMS):

        for i, iseed in enumerate(instance_seeds):
            with nested_run(f"instance_{iseed}", {"instance_seed": iseed}):
                for rseed in run_seeds:
                    r = results[(iseed, rseed)]
                    with nested_run(f"run_{rseed}", {"instance_seed": iseed, "run_seed": rseed}):
                        for layer, ratio in enumerate(r["seed_ratios"], start=1):
                            mlflow.log_metric("approx_ratio", ratio, step=layer)
                        log_metrics_series("optimizer_loss", np.concatenate(r["seed_losses"]))
                        mlflow.log_metric("final_approx_ratio", r["final_approx_ratio"])

        # ── Aggregate ───────────────────────────────────────────────────────────
        all_ratios_arr = np.array(all_ratios)
        finals = np.array(instance_best_ratios)
        n = len(finals)

        mlflow.log_metrics({
            "mean_final_approx_ratio": float(finals.mean()),
            "sem_final_approx_ratio":  float(finals.std(ddof=1) / np.sqrt(n)),
            "max_final_approx_ratio":  float(finals.max()),
            "min_final_approx_ratio":  float(finals.min()),
        })

        log_figure(
            make_ratio_plot(all_ratios_arr, instance_seeds, N_RUNS, N_NODES),
            "approx_ratio_vs_layers.png"
        )
        log_figure(
            make_loss_plot(all_losses, instance_seeds, N_RUNS, N_LAYERS, K_PER_LAYER),
            "optimizer_loss_vs_iterations.png"
        )

    print("\nRun complete. View results with:  mlflow ui")
