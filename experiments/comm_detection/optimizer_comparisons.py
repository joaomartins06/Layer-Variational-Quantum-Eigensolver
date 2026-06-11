import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mlflow
 
from src.community_detection import CommunityDetection
from src.simulator import QuimbSimulator
from src.optimizer import COBYLA, SMO, Adam
from src.lvqe import LayerVQE
from src.logging_utils import start_run, nested_run, log_figure
 
# PYTHONPATH=.

N_NODES        = 8
K_COMMUNITIES  = 4
N_LAYERS       = 2
OPTIMIZERS     = [SMO, COBYLA, Adam]            
OPTIMIZER_KWARGS = {
    "SMO":    {"verbose": True},
    "COBYLA": {"verbose": True},
    "Adam":   {"verbose": True, "lr": 0.15},
}
N_RUNS         = 2
K_PER_LAYER    = 200
K_FINAL        = 500
USE_SAMPLING   = True
N_SAMPLES      = 250
GRAPH_TYPE     = "gnp"
SEED_GRAPH     = 10
SEED_ANGLES    = 50
 
CHECKPOINT_FILE = "experiments/comm_detection/optimizer_comparison_checkpoint.json"
 
PARAMS = dict(
    problem        = "community_detection_optimizer_comparison",
    graph_type     = GRAPH_TYPE,
    num_nodes      = N_NODES,
    k_communities  = K_COMMUNITIES,
    n_layers       = N_LAYERS,
    optimizers     = str([o.__name__ for o in OPTIMIZERS]),
    n_runs         = N_RUNS,
    k_per_layer    = K_PER_LAYER,
    k_final        = K_FINAL,
    use_sampling   = USE_SAMPLING,
    n_samples      = N_SAMPLES,
    seed_graph     = SEED_GRAPH,
    seed_angles    = SEED_ANGLES,
)
 
rng   = np.random.default_rng(SEED_ANGLES)
seeds = rng.integers(0, 10000, size=N_RUNS).tolist()
 
 
# this also needs some sessions, so I need check points
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        print(f"Checkpoint loaded from {CHECKPOINT_FILE}")
        return data
    return {}
 
 
def save_checkpoint(data):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)
    print(f"Checkpoint saved to {CHECKPOINT_FILE}")
 
 
# Helpers
def get_graph(graph_type, N, k, seed):
    if graph_type == "caveman":
        assert N % k == 0
        return nx.connected_caveman_graph(k, N // k)
    elif graph_type == "gnp":
        s = seed
        while True:
            G = nx.gnp_random_graph(N, p=0.15, seed=s)
            if nx.is_connected(G):
                return G
            s += 1
    elif graph_type == "regular":
        s = seed
        while True:
            G = nx.random_regular_graph(3, N, seed=s)
            if nx.is_connected(G):
                return G
            s += 1
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
 
 
def stack_per_layer_losses(losses_per_seed):

    n_seeds    = len(losses_per_seed)
    n_segments = len(losses_per_seed[0])
    means, stds = [], []
    boundaries  = [0]
    for l in range(n_segments):
        min_len = min(len(losses_per_seed[i][l]) for i in range(n_seeds))
        arr = np.array([losses_per_seed[i][l][:min_len] for i in range(n_seeds)])
        means.append(arr.mean(axis=0))
        stds.append(arr.std(axis=0))
        boundaries.append(boundaries[-1] + min_len)
    return np.concatenate(means), np.concatenate(stds), boundaries
 
 
COLORS = ["steelblue", "crimson", "darkorange"]
 
 
def make_loss_plot(checkpoint, optimizers, n_runs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for opt_class, color in zip(optimizers, COLORS):
        entries = checkpoint[opt_class.__name__]
        losses_per_seed = [e["loss_per_layer"] for e in entries.values()]
        mean_c, std_c, boundaries = stack_per_layer_losses(losses_per_seed)
        x = np.arange(len(mean_c))
 
        ax.plot(x, mean_c, color=color, linewidth=1.8, label=opt_class.__name__)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c, color=color, alpha=0.2)
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color=color, linestyle="--", alpha=0.4, linewidth=0.8)
 
    ax.set_xlabel("Total optimisation iterations")
    ax.set_ylabel("Energy (loss)")
    ax.set_title(f"Loss evolution across L-VQE layers — optimizer comparison ({n_runs} seeds)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
 
 
def make_ratio_plot(checkpoint, optimizers, n_layers, n_runs, num_nodes, k, graph_type):
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = np.arange(n_layers + 1)
    for opt_class, color in zip(optimizers, COLORS):
        entries = checkpoint[opt_class.__name__]
        ratios = np.array([e["ratios_per_layer"] for e in entries.values()])
        mean, std = ratios.mean(axis=0), ratios.std(axis=0)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)
        ax.plot(layers, mean, "o-", color=color, linewidth=2, markersize=6,
                label=opt_class.__name__)
 
    ax.set_xlabel("Layer")
    ax.set_ylabel("Approximation ratio")
    ax.set_title(f"L-VQE approximation ratio vs. layers — optimizer comparison\n"
                 f"({n_runs} seeds, {graph_type} graph, n={num_nodes}, k={k})")
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0.5, top=1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
 
 
def make_violin_plot(checkpoint, optimizers, n_layers, n_runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    layers  = np.arange(n_layers + 1)
    n_opts  = len(optimizers)
    width   = 0.8 / n_opts
 
    for opt_idx, (opt_class, color) in enumerate(zip(optimizers, COLORS)):
        entries = checkpoint[opt_class.__name__]
        ratios  = np.array([e["ratios_per_layer"] for e in entries.values()])
        offset  = (opt_idx - (n_opts - 1) / 2) * width
        positions = layers + offset
 
        parts = ax.violinplot(
            [ratios[:, l] for l in range(n_layers + 1)],
            positions=positions, widths=width * 0.9,
            showmedians=True, showextrema=True,
        )
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor("black")
        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in parts:
                parts[key].set_color(color)
 
        ax.plot([], [], color=color, linewidth=8, alpha=0.6, label=opt_class.__name__)
 
    ax.set_xticks(layers)
    ax.set_xticklabels([f"{l} Layer" for l in layers])
    ax.set_ylabel("Approximation ratio")
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f"Approximation ratio distribution — optimizer comparison ({n_runs} seeds)")
    ax.legend()
    ax.grid(True, axis='y', linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# Experiment
sim = QuimbSimulator()
checkpoint = load_checkpoint()

G = get_graph(GRAPH_TYPE, N_NODES, K_COMMUNITIES, seed=SEED_GRAPH)
problem = CommunityDetection(G, k=K_COMMUNITIES, seed=SEED_GRAPH)
q_best  = -problem.best_known_value
print(f"n_qubits: {problem.num_qubits}, n_terms: {len(problem.hamiltonian_terms)}")
print(f"best_known_modularity: {q_best:.4f}")

for opt_class in OPTIMIZERS:
    opt_name = opt_class.__name__
    if opt_name not in checkpoint:
        checkpoint[opt_name] = {}
    opt_kwargs = OPTIMIZER_KWARGS.get(opt_name, {})
 
    for s in seeds:
        if str(s) in checkpoint[opt_name]:
            print(f"Skipping {opt_name} seed={s} (already in checkpoint)")
            continue
 
        print(f"\n Optimizer: {opt_name}  |  Seed: {s}")
        result = LayerVQE(
            problem          = problem,
            simulator        = sim,
            optimizer_class  = opt_class,
            n_layers         = N_LAYERS,
            k_per_layer      = K_PER_LAYER,
            k_final          = K_FINAL,
            use_sampling     = USE_SAMPLING,
            n_samples        = N_SAMPLES,
            record_loss      = True,
            optimizer_kwargs = opt_kwargs,
            seed             = s,
        ).run()
 
        checkpoint[opt_name][str(s)] = {
            "ratios_per_layer":   [float(r) for r in result["history"]["approx_ratio"]],
            "loss_per_layer":     [[float(x) for x in layer_loss]
                                   for layer_loss in result["history"]["optimizer_loss"]],
            "final_approx_ratio": float(result["final_approx_ratio"]),
        }
        save_checkpoint(checkpoint)

 
all_done = all(
    str(s) in checkpoint.get(opt_class.__name__, {})
    for opt_class in OPTIMIZERS
    for s in seeds
)
 
if not all_done:
    remaining = [
        (opt_class.__name__, s) for opt_class in OPTIMIZERS for s in seeds
        if str(s) not in checkpoint.get(opt_class.__name__, {})
    ]
    print(f"\nIncomplete checkpoint. Remaining: {remaining}. Re-run to continue.")
    raise SystemExit


print("\nAll runs complete — logging to MLflow.")

with start_run("lvqe-optimizer-comparison", PARAMS):
    for opt_class in OPTIMIZERS:
        opt_name = opt_class.__name__
        entries  = checkpoint[opt_name]
        ratios_array = np.array([e["ratios_per_layer"] for e in entries.values()])
        finals = ratios_array[:, -1]

        mlflow.log_metrics({
            f"{opt_name}_mean_final_ratio": float(finals.mean()),
            f"{opt_name}_std_final_ratio":  float(finals.std()),
            f"{opt_name}_max_final_ratio":  float(finals.max()),
            f"{opt_name}_min_final_ratio":  float(finals.min()),
        })

        for s in seeds:
            entry = entries[str(s)]
            with nested_run(f"{opt_name}_seed_{s}", {"optimizer": opt_name, "seed": s}):
                for l, r in enumerate(entry["ratios_per_layer"]):
                    mlflow.log_metric("approx_ratio_per_layer", r, step=l)
                mlflow.log_metric("final_approx_ratio", entry["final_approx_ratio"])

    log_figure(make_loss_plot(checkpoint, OPTIMIZERS, N_RUNS),
               "loss_vs_iterations.png")
    log_figure(make_ratio_plot(checkpoint, OPTIMIZERS, N_LAYERS, N_RUNS,
                               N_NODES, K_COMMUNITIES, GRAPH_TYPE),
               "approx_ratio_vs_layers.png")
    log_figure(make_violin_plot(checkpoint, OPTIMIZERS, N_LAYERS, N_RUNS),
               "approx_ratio_violin.png")

print("Done. View results with:  mlflow ui")