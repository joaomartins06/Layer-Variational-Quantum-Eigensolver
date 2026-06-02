"""
Plot final approximation ratio vs. number of layers for Layer VQE and Base VQE
on 52-node 3-regular graphs, pulling data from the MLflow SQLite backend.
"""

import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── MLflow setup ───────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

EXPERIMENT_NAME = "lvqe-maxcut-schwagerl"
N_NODES = 52

LAYER_VQE_SCRIPT = "finite_sampling.py"
BASE_VQE_SCRIPT  = "base_finite_sampling.py"

# ── Load runs ──────────────────────────────────────────────────────────────────
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

all_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=f"params.num_nodes = '{N_NODES}'",
    max_results=200,
)

records = []
for run in all_runs:
    source = run.data.tags.get("mlflow.source.name", "")
    script = os.path.basename(source)
    n_layers = run.data.params.get("n_layers")
    mean_r   = run.data.metrics.get("mean_final_approx_ratio")
    sem_r    = run.data.metrics.get("sem_final_approx_ratio")
    if n_layers is None or mean_r is None:
        continue
    records.append({
        "script":   script,
        "n_layers": int(n_layers),
        "mean":     mean_r,
        "sem":      sem_r if sem_r is not None else 0.0,
        "start_time": run.info.start_time,
    })

df = pd.DataFrame(records)

# Keep only the most recent run per (script, n_layers) pair
df = (
    df.sort_values("start_time", ascending=False)
      .groupby(["script", "n_layers"], as_index=False)
      .first()
)

layer_vqe = df[df["script"] == LAYER_VQE_SCRIPT].sort_values("n_layers")
base_vqe  = df[df["script"] == BASE_VQE_SCRIPT ].sort_values("n_layers")

# The n_layers=0 result is shared — append it to Base VQE if missing
baseline = layer_vqe[layer_vqe["n_layers"] == 0]
if not baseline.empty and 0 not in base_vqe["n_layers"].values:
    base_vqe = pd.concat([baseline, base_vqe], ignore_index=True).sort_values("n_layers")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

COLORS = {"Layer VQE": "steelblue", "Base VQE": "crimson"}

for label, sub_df in [("Layer VQE", layer_vqe), ("Base VQE", base_vqe)]:
    color = COLORS[label]
    x = sub_df["n_layers"].values
    y = sub_df["mean"].values
    e = sub_df["sem"].values

    ax.errorbar(
        x, y, yerr=e,
        fmt="o-", color=color, linewidth=2, markersize=7,
        capsize=4, capthick=1.5, elinewidth=1.5,
        label=label, zorder=3,
    )
    ax.fill_between(x, y - e, y + e, alpha=0.15, color=color, zorder=2)

ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.6, label="Optimal")

ax.set_xlabel("Number of layers", fontsize=12)
ax.set_ylabel("Final approximation ratio", fontsize=12)
ax.set_title(
    f"Layer VQE vs Base VQE — approximation ratio vs. layers\n"
    f"({N_NODES}-node 3-regular graph, mean ± SEM across instances)",
    fontsize=11,
)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylim(bottom=0.85, top=1.05)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.35)
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "model_comparison.pdf")
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
