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
N_NODES = 42
N_LAYERS = 2
SIMULATOR = SMO
N_RUNS = 3
K_PER_LAYER = 10
K_FINAL = 30
USE_SAMPLING = False
N_SAMPLES = 100

PARAMS = dict(
    num_nodes = N_NODES,
    n_layers = N_LAYERS,
    optimizer = SIMULATOR.__name__,
    n_runs = N_RUNS,
    k_per_layer = K_PER_LAYER,
    k_final = K_FINAL,
    use_sampling = USE_SAMPLING,
    n_samples = N_SAMPLES,
)

rng = np.random.default_rng(42)
seeds = rng.integers(0, 10000, size=N_RUNS).tolist()


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


def make_ratio_plot(all_ratios: np.ndarray, n_runs: int, num_nodes: int) -> plt.Figure:
    layers = np.arange(all_ratios.shape[1])
    mean, std = all_ratios.mean(axis=0), all_ratios.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(layers, mean - std, mean + std,
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.plot(layers, mean, "o-", color="steelblue", linewidth=2,
            markersize=6, label="Mean")
    for ratios in all_ratios:
        ax.plot(layers, ratios, color="steelblue", alpha=0.15, linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Approximation ratio")
    ax.set_title(f"L-VQE approximation ratio vs. layers\n"
                 f"(averaged over {n_runs} seeds, {num_nodes}-node 3-regular graph)")
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0.5)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def make_loss_plot(all_losses: list, n_layers: int, k_per_layer: int, n_runs: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, seed_losses in enumerate(all_losses):
        ax.plot(np.concatenate(seed_losses), color="crimson", alpha=0.3,
                linewidth=1.5, label="Seed trajectories" if i == 0 else "")
    for idx in range(1, n_layers + 1):
        ax.axvline(x=k_per_layer * idx, color="black", linestyle="--", alpha=0.6,
                   label="Layer added" if idx == 1 else "")
    ax.set_xlabel("Total optimisation iterations")
    ax.set_ylabel("Energy (loss)")
    ax.set_title(f"Training loss evolution across L-VQE layers\n"
                 f"(showing {n_runs} seeds)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ── Experiment ─────────────────────────────────────────────────────────────────
all_ratios = []
all_losses = []

with start_run("lvqe-maxcut", PARAMS):

    for SEED in seeds:
        print(f"\nRandom seed: {SEED}")
        G = get_random_graph(N_NODES, seed=SEED)
        np.random.seed(SEED)

        result = LayerVQE(
            problem = MaxCut(G, seed=SEED),
            simulator = QuimbSimulator(),
            optimizer_class = SIMULATOR,
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

        with nested_run(f"seed_{SEED}", {"seed": SEED}):
            for layer, ratio in enumerate(seed_ratios, start=1):
                mlflow.log_metric("approx_ratio", ratio, step=layer)
            log_metrics_series("optimizer_loss", np.concatenate(seed_losses))
            mlflow.log_metric("final_approx_ratio", result["final_approx_ratio"])

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_ratios = np.array(all_ratios)
    final_col = all_ratios[:, -1]

    mlflow.log_metrics({
        "mean_final_approx_ratio": float(final_col.mean()),
        "std_final_approx_ratio":  float(final_col.std()),
        "max_final_approx_ratio":  float(final_col.max()),
        "min_final_approx_ratio":  float(final_col.min()),
    })

    log_figure(make_ratio_plot(all_ratios, N_RUNS, N_NODES),
               "approx_ratio_vs_layers.pdf")
    log_figure(make_loss_plot(all_losses, N_LAYERS, K_PER_LAYER, N_RUNS),
               "optimizer_loss_vs_iterations.pdf")

print("\nRun complete. View results with:  mlflow ui")