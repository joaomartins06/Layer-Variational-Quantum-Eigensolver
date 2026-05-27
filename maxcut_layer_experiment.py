import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.maxcut import MaxCut
from src.simulator import QuimbSimulator
from src.optimizer import COBYLA, SMO
from src.lvqe import LayerVQE
from collections import Counter

num_nodes = 42
N_LAYERS = 2
SIMULATOR=SMO        # SMO or COBYLA
N_RUNS=5
K_PER_LAYER=200
K_FINAL=1500

rng = np.random.default_rng(42)
seeds = rng.integers(0, 10000, size=N_RUNS).tolist()

def get_random_graph(N: int, seed: int = 42, plot=False):
    assert N % 2 == 0
    while True:
        G = nx.random_regular_graph(3, num_nodes)
        if nx.is_connected(G):
            if plot:
                plt.figure(figsize=(8, 6))
                pos = nx.spring_layout(G, seed=seed)
                nx.draw(G, pos, with_labels=True, node_color='lightblue',
                    node_size=400, font_size=10, font_weight='bold')
                plt.title(f"Random regular graph G(k=3, N={G.number_of_nodes()}): {G.number_of_edges()} edges")
                plt.show()
            return G
        else:
            seed += 1
    return None

all_ratios = []
all_losses = []

for SEED in seeds:
    print(f"\nRandom seed: {SEED}")

    G = get_random_graph(num_nodes, seed=SEED, plot=False)

    np.random.seed(SEED)

    problem = MaxCut(G, seed=SEED)

    sim = QuimbSimulator()

    # run L-VQE
    lvqe = LayerVQE(
        problem=problem,
        simulator=sim,
        optimizer_class=SIMULATOR,
        n_layers=N_LAYERS,
        k_per_layer=K_PER_LAYER,
        k_final=K_FINAL,
        use_sampling=True,
        n_samples=100,
        record_loss=True
    )

    result = lvqe.run()

    seed_ratios = []
    for layer in result['history']['layer']:
        ratio = result['history']['approx_ratio'][layer]
        seed_ratios.append(ratio)
        print(f" Approximation ratio after layer {layer}/{N_LAYERS} : {ratio}")

    all_ratios.append(seed_ratios)
    all_losses.append(result['history']['optimizer_loss'])
    print(f"\nFinal approximation ratio: {result['final_approx_ratio']:.4f}\n")

# --- Plot ---
all_ratios = np.array(all_ratios)          # (n_seeds, n_checkpoints)
n_checkpoints = all_ratios.shape[1]
layers = np.arange(n_checkpoints)
mean = all_ratios.mean(axis=0)
std  = all_ratios.std(axis=0)

fig, ax = plt.subplots(figsize=(7, 4))

ax.fill_between(layers, mean - std, mean + std, alpha=0.25, color='steelblue', label='±1 std')
ax.plot(layers, mean, 'o-', color='steelblue', linewidth=2, markersize=6, label='Mean')

for i, seed_ratios in enumerate(all_ratios):
    ax.plot(layers, seed_ratios, color='steelblue', alpha=0.15, linewidth=1)

ax.set_xlabel('Layer')
ax.set_ylabel('Approximation ratio')
ax.set_title(f'L-VQE approximation ratio vs. layers\n(averaged over {len(seeds)} seeds, {num_nodes}-node 3-regular graph)')
ax.set_xticks(layers)
ax.set_ylim(bottom=0.5)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
fig.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))

# We will plot the continuous loss curve for each seed
for i, seed_losses in enumerate(all_losses):
    # Flatten the list of arrays into a single continuous training curve
    continuous_loss = np.concatenate(seed_losses)
    ax2.plot(continuous_loss, color='crimson', alpha=0.3, linewidth=1.5,
             label='Seed Trajectories' if i == 0 else "")

# Calculate transition points (where a new layer is added)
# Using the first seed as a reference for lengths
if all_losses:
    transition_points = [K_PER_LAYER * layer for layer in range(1,N_LAYERS+1)]

    for idx, pt in enumerate(transition_points):
        ax2.axvline(x=pt, color='black', linestyle='--', alpha=0.6,
                    label='Layer Added' if idx == 0 else "")

ax2.set_xlabel('Total Optimization Iterations')
ax2.set_ylabel('Energy (Loss)')
ax2.set_title(f'Training Loss Evolution Across L-VQE Layers\n(Showing {len(seeds)} seeds)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)
fig2.tight_layout()
plt.show()