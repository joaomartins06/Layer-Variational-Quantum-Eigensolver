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
N_NODES = 32
N_INSTANCES = 1
N_RUNS = 1

N_LAYERS =1
K_PER_LAYER = 50
K_FINAL = 1000

SIMULATOR = SMO

USE_SAMPLING = True
N_SAMPLES = 100

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

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, seed in enumerate(instance_seeds):
        instance_ratios = all_ratios[i * n_runs : (i + 1) * n_runs]  # [n_runs, n_layers]
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
    all_losses: list,           # shape: [n_instances * n_runs], each entry is a list of per-layer loss arrays
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

                result = LayerVQE(
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

                seed_ratios = [result["history"]["approx_ratio"][l]
                               for l in result["history"]["layer"]]
                seed_losses = result["history"]["optimizer_loss"]

                for layer, ratio in enumerate(seed_ratios, start=1):
                    print(f"  Approx ratio after layer {layer}/{N_LAYERS}: {ratio:.4f}")
                print(f"  Final approx ratio: {result['final_approx_ratio']:.4f}")

                all_ratios.append(seed_ratios)
                all_losses.append(seed_losses)
                print(len(seed_losses))
                instance_run_finals.append(result["final_approx_ratio"])

                with nested_run(f"run_{RUN_SEED}", {"instance_seed": INSTANCE_SEED, "run_seed": RUN_SEED}):
                    for layer, ratio in enumerate(seed_ratios, start=1):
                        mlflow.log_metric("approx_ratio", ratio, step=layer)
                    log_metrics_series("optimizer_loss", np.concatenate(seed_losses))
                    mlflow.log_metric("final_approx_ratio", result["final_approx_ratio"])

        instance_best_ratios.append(max(instance_run_finals))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_ratios = np.array(all_ratios)
    finals = np.array(instance_best_ratios)

    n = len(finals)
    mean = float(finals.mean())
    #sem = float(finals.std(ddof=1) / np.sqrt(n))

    mlflow.log_metrics({
        "mean_final_approx_ratio": mean,
        #"sem_final_approx_ratio": sem,
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