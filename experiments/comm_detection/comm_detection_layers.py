import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mlflow
import matplotlib.cm as cm
from networkx.algorithms.community import louvain_communities
from sklearn.neighbors import kneighbors_graph

from src.community_detection import CommunityDetection
from src.simulator import QuimbSimulator
from src.optimizer import COBYLA, SMO, Adam
from src.lvqe import LayerVQE
from src.logging_utils import start_run, nested_run, log_figure, log_metrics_series

#sorry, this is the command I have to add to run, just so I dont forget :)

#PYTHONPATH=.


# ── Config ─────────────────────────────────────────────────────────────────────
N_NODES = 20
K_COMMUNITIES = 4
N_LAYERS = 2
OPTIMIZER = COBYLA
N_RUNS = 5
K_PER_LAYER = 200
K_FINAL = 500
USE_SAMPLING = False
N_SAMPLES = 250
GRAPH_TYPE = "caveman"   # "caveman", "gnp", "regular", "gaussian", "windmill"
SEED_GRAPH = 10
SEED_ANGLES = 50
LEARNING_RATE = 0.15

PARAMS = dict(
    problem = "community_detection",
    graph_type = GRAPH_TYPE,
    num_nodes = N_NODES,
    k_communities = K_COMMUNITIES,
    n_layers = N_LAYERS,
    optimizer = OPTIMIZER.__name__,
    n_runs = N_RUNS,
    k_per_layer = K_PER_LAYER,
    k_final = K_FINAL,
    use_sampling = USE_SAMPLING,
    n_samples = N_SAMPLES,
    seed_graph = SEED_GRAPH,
    seed_angles = SEED_ANGLES,
    learning_rate = LEARNING_RATE,
)

rng = np.random.default_rng(SEED_ANGLES)
seeds = rng.integers(0, 10000, size=N_RUNS).tolist()


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_graph(graph_type: str, N: int, k: int, seed: int):
    """Build a graph of the requested type, ensuring connectivity."""
    if graph_type == "caveman":
        # k cliques of size N//k, connected by single bridges
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
    
    elif graph_type == "gaussian":
        n_per_cluster = N // k
        std = 0.5
        centres = [(0, 0), (3, 0), (0, 3), (3, 3)][:k]

        s = seed
        while True:
            rng = np.random.default_rng(s)
            pos_array = np.vstack([
                rng.normal(loc=c, scale=std, size=(n_per_cluster, 2))
                for c in centres
            ])
            A = kneighbors_graph(pos_array, n_neighbors=3, mode='connectivity', include_self=False)
            A = (A + A.T)
            G = nx.from_scipy_sparse_array(A)
            if nx.is_connected(G):
                return G
            s += 1
    elif graph_type == "windmill":
        n_cliques, clique_size = 4, 5  # 17 nodes
        return nx.windmill_graph(n_cliques, clique_size)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def decode_by_sampling(sim, params, ansatz, problem, n_samples=2000):
    #realist decoding of the final state.
    #sample bitstrings from the final circuit, pick the most probable assignment, 
    #compute its true modularity.
    bitstrings = sim.get_most_frequent_assignments(
        params, ansatz, problem=problem, n_samples=n_samples
    )
    best_assignment, best_proba = bitstrings[0]
    modularity = problem.evaluate(best_assignment)
    q_best = -problem.best_known_value
    return modularity, modularity / q_best, best_assignment, best_proba

def make_partition_plot(G, lvqe_assignment, lvqe_modularity,
                        best_assignment, best_modularity, seed: int) -> plt.Figure:
    """Side-by-side comparison: L-VQE decoded partition vs Louvain best-known."""
    pos = nx.spring_layout(G, seed=42)

    def draw(ax, assignment, title):
        n_colors = max(assignment) + 1
        cmap = cm.get_cmap("tab10", max(n_colors, 2))
        node_colors = [cmap(assignment[node]) for node in G.nodes()]
        nx.draw_networkx(
            G, pos=pos, node_color=node_colors, with_labels=True,
            node_size=600, font_color="white", font_weight="bold",
            edge_color="gray", ax=ax,
        )
        ax.set_title(title)
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw(axes[0], lvqe_assignment, f"L-VQE decoded (seed {seed}, Q = {lvqe_modularity:.4f})")
    draw(axes[1], best_assignment, f"Best known / Louvain (Q = {best_modularity:.4f})")
    fig.tight_layout()
    return fig


def make_ratio_plot(all_ratios: np.ndarray, n_runs: int, num_nodes: int,
                    k: int, graph_type: str) -> plt.Figure:
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
    ax.set_ylabel("Approximation ratio (energy-based)")
    ax.set_title(f"L-VQE approximation ratio vs. layers\n"
                 f"({n_runs} seeds, {graph_type} graph, n={num_nodes}, k={k})")
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0.5, top=1.05)
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
    ax.set_title(f"Training loss evolution across L-VQE layers ({n_runs} seeds)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ── Experiment ─────────────────────────
all_ratios_energy = []
all_losses = []
all_modularities_decoded = []
all_ratios_decoded = []
all_assignments = []  

sim = QuimbSimulator()

with start_run("lvqe-comm-detection", PARAMS):

    #notice that the only thing changing between runs is the angle initialization
    #the graph/problem remains the same throughout the runs
    
    G = get_graph(GRAPH_TYPE, N_NODES, K_COMMUNITIES, seed=SEED_GRAPH)
    problem = CommunityDetection(G, k=K_COMMUNITIES, seed=SEED_GRAPH)
    q_best = -problem.best_known_value
    print(f"n_qubits: {problem.num_qubits}, n_terms: {len(problem.hamiltonian_terms)}")
    print(f"best_known_modularity: {q_best:.4f}") 

    for s in seeds:
        np.random.seed(s)
        print(f"\nRandom seed: {s}")
        result = LayerVQE(
            problem = problem,
            simulator = sim,
            optimizer_class = OPTIMIZER,
            #optimizer_kwargs={"lr": LEARNING_RATE},
            n_layers = N_LAYERS,
            k_per_layer = K_PER_LAYER,
            k_final = K_FINAL,
            use_sampling = USE_SAMPLING,
            n_samples = N_SAMPLES,
            record_loss = True,
        ).run()

        seed_ratios_energy = result["history"]["approx_ratio"]
        seed_losses = result["history"]["optimizer_loss"]
        

        #final realistic measurement: sample the optimised circuit
        modularity_decoded, ratio_decoded, best_assignment, best_proba = decode_by_sampling(
            sim, result["final_params"], result["final_ansatz"], problem, n_samples=N_SAMPLES
        )

        all_assignments.append((s, modularity_decoded, best_assignment))

        for layer, ratio in enumerate(seed_ratios_energy):
            print(f"Approx ratio (energy) after layer {layer}: {ratio:.4f}")
        print(f"Final approx ratio (energy):  {result['final_approx_ratio']:.4f}")
        print(f"Final approx ratio (decoded): {ratio_decoded:.4f}")
        print(f"Final modularity (decoded):   {modularity_decoded:.4f}")
        print(f"Most probable bitstring prob: {best_proba:.2f}%")

        all_ratios_energy.append(seed_ratios_energy)
        all_losses.append(seed_losses)
        all_modularities_decoded.append(modularity_decoded)
        all_ratios_decoded.append(ratio_decoded)

        with nested_run(f"seed_{s}", {"seed": s}):
            for layer, ratio in enumerate(seed_ratios_energy):
                mlflow.log_metric("approx_ratio_energy", ratio, step=layer)
            log_metrics_series("optimizer_loss", np.concatenate(seed_losses))
            mlflow.log_metric("final_approx_ratio_energy", result["final_approx_ratio"])
            mlflow.log_metric("final_approx_ratio_decoded", ratio_decoded)
            mlflow.log_metric("final_modularity_decoded", modularity_decoded)
            mlflow.log_metric("best_known_modularity", q_best)
            mlflow.log_metric("best_bitstring_probability_pct", best_proba)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_ratios_energy = np.array(all_ratios_energy)
    final_col = all_ratios_energy[:, -1]
    decoded = np.array(all_ratios_decoded)
    mods = np.array(all_modularities_decoded)

    mlflow.log_metrics({
        "mean_final_approx_ratio_energy":  float(final_col.mean()),
        "std_final_approx_ratio_energy":   float(final_col.std()),
        "mean_final_approx_ratio_decoded": float(decoded.mean()),
        "std_final_approx_ratio_decoded":  float(decoded.std()),
        "max_final_approx_ratio_decoded":  float(decoded.max()),
        "min_final_approx_ratio_decoded":  float(decoded.min()),
        "mean_final_modularity_decoded":   float(mods.mean()),
    })

    log_figure(make_ratio_plot(all_ratios_energy, N_RUNS, N_NODES, K_COMMUNITIES, GRAPH_TYPE),
               "approx_ratio_vs_layers.png")
    log_figure(make_loss_plot(all_losses, N_LAYERS, K_PER_LAYER, N_RUNS),
               "optimizer_loss_vs_iterations.png")
    
    #pick the best run across all seeds (highest decoded modularity)
    best_seed, best_mod_lvqe, best_assignment_lvqe = max(all_assignments, key=lambda x: x[1])

    #louvain reference partition
    best_assignment_louvain = [0] * problem.num_nodes
    for comm_idx, comm in enumerate(louvain_communities(G, seed=SEED_GRAPH)):
        for node in comm:
            best_assignment_louvain[node] = comm_idx
    best_mod_louvain = problem.evaluate(best_assignment_louvain)

    log_figure(make_partition_plot(G, best_assignment_lvqe, best_mod_lvqe,
                            best_assignment_louvain, best_mod_louvain, best_seed),
                "best_partition_comparison.png")

print("\nRun complete. View results with:  mlflow ui")