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
 
N_NODES_LIST   = [8, 12, 16, 20]
K_COMMUNITIES  = 4
LAYER_CONFIGS  = [0, 1, 2] 

OPTIMIZER      = COBYLA
N_RUNS         = 3
K_PER_LAYER    = 200
MAX_ITER       = 2400
EPSILON        = 0.5e-2
T_MAX          = 100
USE_SAMPLING   = False
N_SAMPLES      = 200
LEARNING_RATE  = 0.1

GRAPH_TYPE     = "gnp"
SEED_GRAPH     = 10
SEED_ANGLES    = 50

CHECKPOINT_FILE = "experiments/comm_detection/convergence_checkpoint.json"
 
PARAMS = dict(
    problem        = "community_detection_scaling",
    graph_type     = GRAPH_TYPE,
    n_nodes_range  = f"{N_NODES_LIST[0]}-{N_NODES_LIST[-1]}",
    k_communities  = K_COMMUNITIES,
    layer_configs  = str(LAYER_CONFIGS),
    optimizer      = OPTIMIZER.__name__,
    n_runs         = N_RUNS,
    k_per_layer    = K_PER_LAYER,
    max_iter       = MAX_ITER,
    epsilon        = EPSILON,
    t_max          = T_MAX,
    use_sampling   = USE_SAMPLING,
    n_samples      = N_SAMPLES,
    seed_graph     = SEED_GRAPH,
    seed_angles    = SEED_ANGLES,
    learning_rate  = LEARNING_RATE,
)
 
rng   = np.random.default_rng(SEED_ANGLES)
seeds = rng.integers(0, 10000, size=N_RUNS).tolist()
 
 
#This experiment is very long, so I had to run it in more than one session :/
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            raw = json.load(f)
        results       = {nl: {int(n): v for n, v in raw["results"][str(nl)].items()}       for nl in LAYER_CONFIGS}
        conv_flags    = {nl: {int(n): v for n, v in raw["conv_flags"][str(nl)].items()}    for nl in LAYER_CONFIGS}
        approx_ratios = {nl: {int(n): v for n, v in raw["approx_ratios"][str(nl)].items()} for nl in LAYER_CONFIGS}
        print(f"Checkpoint loaded from {CHECKPOINT_FILE}")
        return results, conv_flags, approx_ratios
    return (
        {nl: {} for nl in LAYER_CONFIGS},
        {nl: {} for nl in LAYER_CONFIGS},
        {nl: {} for nl in LAYER_CONFIGS},
    )
 
 
def save_checkpoint(results, conv_flags, approx_ratios):
    data = {
        "results":       {str(nl): {str(n): v for n, v in d.items()} for nl, d in results.items()},
        "conv_flags":    {str(nl): {str(n): v for n, v in d.items()} for nl, d in conv_flags.items()},
        "approx_ratios": {str(nl): {str(n): v for n, v in d.items()} for nl, d in approx_ratios.items()},
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)
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
 
 
def find_convergence_iteration(loss_history, epsilon, t_max):
    arr = np.asarray(loss_history, dtype=float)
    threshold  = arr.min() + epsilon
    violations = np.where(arr > threshold)[0]

    if len(violations) == 0:
        conv_iter = 0
    else:
        conv_iter = int(violations[-1]) + 1

    if conv_iter + t_max > len(arr):
        return False, len(arr)
    return True, conv_iter
 
 
def power_law_fit(n_arr, iter_mean, iter_std):
    if len(n_arr) < 3:
        return np.nan, np.nan, np.nan, np.nan
    log_n       = np.log2(n_arr)
    log_y       = np.log2(iter_mean)
    sigma_log_y = np.maximum(iter_std / (iter_mean * np.log(2)), 1e-12)
    all_zero    = np.all(iter_std < 1e-12)
    weights     = None if all_zero else 1.0 / sigma_log_y
    coeffs, cov = np.polyfit(log_n, log_y, deg=1, w=weights, cov=True)
    a, b        = coeffs
    return float(a), float(np.sqrt(cov[0, 0])), float(b), float(np.sqrt(cov[1, 1]))
 
 
def make_scaling_plot(results, layer_configs, graph_type, optimizer_name, use_sampling):
    colors = ["purple", "teal", "gold"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
 
    for nl, color in zip(layer_configs, colors):
        n_arr     = np.array(sorted(results[nl].keys()))
        iter_mean = np.array([np.mean(results[nl][n]) for n in n_arr])
        iter_std  = np.array([np.std(results[nl][n])  for n in n_arr])
 
        a, sa, b, sb = power_law_fit(n_arr, iter_mean, iter_std)
 
        if np.isnan(a):
            ax.errorbar(n_arr, iter_mean, yerr=iter_std, fmt="o-", color=color,
                        capsize=3, linewidth=1.5, label=f"{nl} Layer (insufficient points for fit)")
        else:
            ax.errorbar(n_arr, iter_mean, yerr=iter_std, fmt="o-", color=color,
                        capsize=3, linewidth=1.5,
                        label=f"{nl} Layer  log(y) = ({a:.2f}±{sa:.2f}) log(n) + ({b:.2f}±{sb:.2f})")
            n_fine = np.linspace(n_arr.min(), n_arr.max(), 200)
            ax.plot(n_fine, 2**b * n_fine**a, "--", color=color, alpha=0.6)
 
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel("Iterations (y)")
    ax.set_title(f"Scaling analysis of L-VQE on log-log scale\n"
                 f"({graph_type} graph, {optimizer_name}, "
                 f"{'sampling' if use_sampling else 'exact'}, {N_RUNS} seeds)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
 
 
def log_all_to_mlflow(results, conv_flags, approx_ratios):
    with start_run("lvqe-scaling-analysis", PARAMS):
        for n_nodes in N_NODES_LIST:
            for n_layers in LAYER_CONFIGS:
                iters = np.array(results[n_layers][n_nodes])
                mlflow.log_metric(f"mean_iters_L{n_layers}",        float(iters.mean()),                           step=n_nodes)
                mlflow.log_metric(f"std_iters_L{n_layers}",         float(iters.std()),                            step=n_nodes)
                mlflow.log_metric(f"conv_rate_L{n_layers}",         float(np.mean(conv_flags[n_layers][n_nodes])), step=n_nodes)
                mlflow.log_metric(f"mean_approx_ratio_L{n_layers}", float(np.mean(approx_ratios[n_layers][n_nodes])), step=n_nodes)
 
                for s, n_iter, converged, ratio in zip(
                    seeds,
                    results[n_layers][n_nodes],
                    conv_flags[n_layers][n_nodes],
                    approx_ratios[n_layers][n_nodes],
                ):
                    with nested_run(f"n{n_nodes}_L{n_layers}_seed{s}",
                                    {"n_nodes": n_nodes, "n_layers": n_layers, "seed": s}):
                        mlflow.log_metric("iters_to_convergence", n_iter)
                        mlflow.log_metric("converged",            int(converged))
                        mlflow.log_metric("final_approx_ratio",   ratio)
 
        fig = make_scaling_plot(results, LAYER_CONFIGS, GRAPH_TYPE,
                                OPTIMIZER.__name__, USE_SAMPLING)
        log_figure(fig, "scaling_iterations_vs_nodes.png")
 
        for nl in LAYER_CONFIGS:
            n_arr     = np.array(sorted(results[nl].keys()))
            iter_mean = np.array([np.mean(results[nl][n]) for n in n_arr])
            iter_std  = np.array([np.std(results[nl][n])  for n in n_arr])
            a, sa, b, sb = power_law_fit(n_arr, iter_mean, iter_std)
            if not np.isnan(a):
                mlflow.log_metric(f"fit_slope_L{nl}",         a)
                mlflow.log_metric(f"fit_slope_err_L{nl}",     sa)
                mlflow.log_metric(f"fit_intercept_L{nl}",     b)
                mlflow.log_metric(f"fit_intercept_err_L{nl}", sb)
                print(f"  L={nl}:  a = {a:.3f} ± {sa:.3f},   b = {b:.3f} ± {sb:.3f}")
 
 
# Experiment
sim = QuimbSimulator()
 
results, conv_flags, approx_ratios = load_checkpoint()
 
optimizer_kwargs = {"verbose": True}
if OPTIMIZER is Adam:
    optimizer_kwargs["lr"] = LEARNING_RATE
 
for n_nodes in N_NODES_LIST:
    if all(n_nodes in results[nl] for nl in LAYER_CONFIGS):
        print(f"Skipping n={n_nodes} (already in checkpoint)")
        continue
 
    print(f"\n{'='*72}\n n_nodes = {n_nodes}\n{'='*72}")
 
    G = get_graph(GRAPH_TYPE, n_nodes, K_COMMUNITIES, seed=SEED_GRAPH)
    problem = CommunityDetection(G, k=K_COMMUNITIES, seed=SEED_GRAPH)
    print(f"  n_qubits: {problem.num_qubits}, n_terms: {len(problem.hamiltonian_terms)}")
 
    for nl in LAYER_CONFIGS:
        results[nl][n_nodes]       = []
        conv_flags[nl][n_nodes]    = []
        approx_ratios[nl][n_nodes] = []
 
    for s in seeds:
        print(f"\n  --- seed = {s} ---")
        for n_layers in LAYER_CONFIGS:
            k_per_run = MAX_ITER if n_layers == 0 else K_PER_LAYER
 
            result = LayerVQE(
                problem          = problem,
                simulator        = sim,
                optimizer_class  = OPTIMIZER,
                n_layers         = n_layers,
                k_per_layer      = k_per_run,
                k_final          = MAX_ITER,
                use_sampling     = USE_SAMPLING,
                n_samples        = N_SAMPLES,
                record_loss      = True,
                optimizer_kwargs = optimizer_kwargs,
                seed             = s,
            ).run()
 
            all_layers_loss   = np.concatenate(result["history"]["optimizer_loss"])
            converged, n_iter = find_convergence_iteration(
                all_layers_loss, EPSILON, T_MAX
            )
            ratio = result["final_approx_ratio"]
 
            results[n_layers][n_nodes].append(n_iter)
            conv_flags[n_layers][n_nodes].append(converged)
            approx_ratios[n_layers][n_nodes].append(ratio)
 
            tag = "OK" if converged else "CAP"
            print(f"    L={n_layers}: iters={n_iter:>5} [{tag}], approx_ratio={ratio:+.4f}")
 
    save_checkpoint(results, conv_flags, approx_ratios)
 
# ── MLflow + plot — only when all nodes are done ───────────────────────────────
all_done = all(
    n_nodes in results[nl]
    for n_nodes in N_NODES_LIST
    for nl in LAYER_CONFIGS
)
 
if all_done:
    print("\nAll nodes complete — logging to MLflow.")
    log_all_to_mlflow(results, conv_flags, approx_ratios)
    print("Done. View results with:  mlflow ui")
else:
    completed = [n for n in N_NODES_LIST if all(n in results[nl] for nl in LAYER_CONFIGS)]
    remaining = [n for n in N_NODES_LIST if n not in completed]
    print(f"\nProgress: {completed} done, {remaining} remaining. Re-run to continue.")