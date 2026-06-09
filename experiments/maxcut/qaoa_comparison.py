import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mlflow

from src.maxcut import MaxCut
from src.simulator import QuimbSimulator
from src.optimizer import COBYLA, SMO
from src.lvqe import LayerVQE
from src.logging_utils import start_run, nested_run, log_figure, log_metrics_series



# ── Config ─────────────────────────────────────────────────────────────────────
N_NODES = 52
N_INSTANCES = 10
N_RUNS = 5

# L-VQE
N_LAYERS = 2
K_PER_LAYER = 50
K_FINAL = 400
SIMULATOR = SMO
USE_SAMPLING = False
N_SAMPLES = 100

# QAOA


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
def get_random_graph(N: int, seed: int, plot: bool = False):
    assert N % 2 == 0
    while True:
        G = nx.random_regular_graph(3, N)
        if nx.is_connected(G):
            if plot:
                plt.figure(figsize=(8, 6))
                pos = nx.spring_layout(G, seed=seed)
                nx.draw(G, pos, with_labels=True, node_color="lightblue",
                        node_size=400, font_size=10, font_weight="bold")
                plt.title(f"Random regular graph G(k=3, N={G.number_of_nodes()}): "
                          f"{G.number_of_edges()} edges")
                plt.show()
            return G
        else:
            seed += 1


def make_ratio_plot(
    all_ratios: np.ndarray,   # shape: [n_instances * n_runs, n_layers]
    instance_seeds: list,
    n_runs: int,
    num_nodes: int,
) -> plt.Figure:
    n_instances = len(instance_seeds)
    colors = plt.cm.tab10(np.linspace(0, 1, n_instances))
    n_layers = all_ratios.shape[1]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, seed in enumerate(instance_seeds):
        instance_ratios = all_ratios[i * n_runs : (i + 1) * n_runs]  # [n_runs, n_layers]
        mean = instance_ratios.mean(axis=0)
        std  = instance_ratios.std(axis=0)

        ax.fill_between(layers, mean - std, mean + std,
                        alpha=0.10, color=colors[i])
        ax.plot(layers, mean, "o-", color=colors[i], linewidth=2,
                markersize=6, label=f"Instance seed {seed}")
        for ratios in instance_ratios:
            ax.plot(layers, ratios, color=colors[i], alpha=0.15, linewidth=1)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Approximation ratio")
    ax.set_title(
        f"L-VQE approximation ratio vs. layers\n"
        f"(mean ± std over {n_runs} runs, {n_instances} instances, {num_nodes}-node 3-regular graph)"
    )
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0.5)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def make_loss_plot(
    all_losses: list,           # shape: [n_instances * n_runs], each entry is a list of per-layer loss arrays
    instance_seeds: list,
    n_runs: int,
    n_layers: int,
    k_per_layer: int,
) -> plt.Figure:
    n_instances = len(instance_seeds)
    colors = plt.cm.tab10(np.linspace(0, 1, n_instances))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, seed in enumerate(instance_seeds):
        # Grab the n_runs loss trajectories for this instance
        instance_losses = all_losses[i * n_runs : (i + 1) * n_runs]
        trajectories = np.array([np.concatenate(run) for run in instance_losses])

        mean = trajectories.mean(axis=0)
        std  = trajectories.std(axis=0)
        xs   = np.arange(len(mean))

        ax.fill_between(xs, mean - std, mean + std,
                        alpha=0.10, color=colors[i])
        ax.plot(xs, mean, linewidth=2, color=colors[i],
                label=f"Instance seed {seed}")

    for idx in range(1, n_layers + 1):
        ax.axvline(x=k_per_layer * idx, color="black",
                   linestyle="--", alpha=0.5,
                   label="Layer added" if idx == 1 else "")

    ax.set_xlabel("Total optimisation iterations")
    ax.set_ylabel("Energy (loss)")
    ax.set_title(
        f"Training loss per instance (mean ± std over {n_runs} runs)\n"
        f"{n_instances} instances, {n_layers} layers"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig





# ── Experiment ─────────────────────────────────────────────────────────────────
all_ratios = []
all_losses = []
instance_best_ratios = []

with start_run("lvqe-maxcut-schwagerl", PARAMS):

    for i, INSTANCE_SEED in enumerate(instance_seeds):
        print(f"\n{'=' * 60}")
        print(f"INSTANCE {i + 1}/{N_INSTANCES}  (seed={INSTANCE_SEED})")
        print(f"{'=' * 60}")
        G = get_random_graph(N_NODES, seed=INSTANCE_SEED)

        instance_run_finals = []

        with nested_run(f"instance_{INSTANCE_SEED}", {"instance_seed": INSTANCE_SEED}):
            for j, RUN_SEED in enumerate(run_seeds):
                print(f"\n  ── Run {j + 1}/{N_RUNS}  (seed={RUN_SEED})")

                np.random.seed(RUN_SEED)

                # L-VQE

                result_lvqe = LayerVQE(
                    problem = MaxCut(G, seed=INSTANCE_SEED),
                    simulator = QuimbSimulator(),
                    optimizer_class = SIMULATOR,
                    seed=RUN_SEED,
                    n_layers = N_LAYERS,
                    k_per_layer = K_PER_LAYER,
                    k_final = K_FINAL,
                    use_sampling = USE_SAMPLING,
                    n_samples = N_SAMPLES,
                    record_loss = True,
                ).run()

                seed_ratios = [result_lvqe["history"]["approx_ratio"][l]
                               for l in result_lvqe["history"]["layer"]]
                seed_losses = result_lvqe["history"]["optimizer_loss"]

                for layer, ratio in enumerate(seed_ratios, start=1):
                    print(f"  Approx ratio after layer {layer}/{N_LAYERS}: {ratio:.4f}")
                print(f"  Final approx ratio: {result_lvqe['final_approx_ratio']:.4f}")

                all_ratios.append(seed_ratios)
                all_losses.append(seed_losses)
                print(len(seed_losses))
                instance_run_finals.append(result_lvqe["final_approx_ratio"])

                # QAOA

                with nested_run(f"run_{RUN_SEED}", {"instance_seed": INSTANCE_SEED, "run_seed": RUN_SEED}):
                    for layer, ratio in enumerate(seed_ratios, start=1):
                        mlflow.log_metric("approx_ratio", ratio, step=layer)
                    log_metrics_series("optimizer_loss", np.concatenate(seed_losses))
                    mlflow.log_metric("final_approx_ratio", result_lvqe["final_approx_ratio"])

        instance_best_ratios.append(max(instance_run_finals))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_ratios = np.array(all_ratios)
    finals = np.array(instance_best_ratios)

    n = len(finals)
    mean = float(finals.mean())
    sem = float(finals.std(ddof=1) / np.sqrt(n))

    mlflow.log_metrics({
        "mean_final_approx_ratio": mean,
        "sem_final_approx_ratio": sem,
        "max_final_approx_ratio": float(finals.max()),
        "min_final_approx_ratio": float(finals.min()),
    })

    log_figure(
        make_ratio_plot(all_ratios, instance_seeds, N_RUNS, N_NODES),
        "approx_ratio_vs_layers.png"
    )
    log_figure(
        make_loss_plot(all_losses, instance_seeds, N_RUNS, N_LAYERS, K_PER_LAYER),
        "optimizer_loss_vs_iterations.png"
    )

print("\nRun complete. View results with:  mlflow ui")