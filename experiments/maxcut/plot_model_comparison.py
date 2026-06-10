"""
Plot final approximation ratio vs. number of layers for Layer VQE and Base VQE
on 52-node 3-regular graphs, pulling data from the MLflow SQLite backend.

Runs are uploaded via log_results.py from GitHub Actions JSON artifacts.
Run names follow the pattern "layer{N}" / "base{N}" — model type and layer count
are parsed directly from the name.  The n_layers=0 point (base0) is shared
between both curves.
"""

import os
import re
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── MLflow setup ───────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "../../mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

EXPERIMENT_NAME = "lvqe-maxcut-schwagerl"
N_NODES = 52

# ── Load runs ──────────────────────────────────────────────────────────────────
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

all_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    max_results=500,
)

NAME_RE = re.compile(r"^(layer|base)(\d+)$")

records = []
for run in all_runs:
    name = run.data.tags.get("mlflow.runName", "")
    m = NAME_RE.match(name)
    if not m:
        continue
    mean_r = run.data.metrics.get("mean_final_approx_ratio")
    sem_r  = run.data.metrics.get("sem_final_approx_ratio")
    if mean_r is None:
        continue
    model = "Layer VQE" if m.group(1) == "layer" else "Base VQE"
    records.append({
        "model":      model,
        "n_layers":   int(m.group(2)),
        "mean":       mean_r,
        "sem":        sem_r if sem_r is not None else 0.0,
        "start_time": run.info.start_time,
    })

df = pd.DataFrame(records)

# Keep only the most recent run per (model, n_layers) pair
df = (
    df.sort_values("start_time", ascending=False)
      .groupby(["model", "n_layers"], as_index=False)
      .first()
)

layer_vqe = df[df["model"] == "Layer VQE"].sort_values("n_layers")
base_vqe  = df[df["model"] == "Base VQE" ].sort_values("n_layers")

# n_layers=0 (base0) is shared — add it to the Layer VQE curve too
baseline = base_vqe[base_vqe["n_layers"] == 0]
if not baseline.empty and 0 not in layer_vqe["n_layers"].values:
    layer_vqe = pd.concat(
        [baseline.assign(model="Layer VQE"), layer_vqe], ignore_index=True
    ).sort_values("n_layers")

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

ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.6, label="BKV (SW)")

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

out_path = os.path.join(os.path.dirname(__file__), "results/model_comparison.pdf")
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()

# ── Beamer-ready figure ────────────────────────────────────────────────────────
import matplotlib as mpl

with mpl.rc_context({"font.size": 13, "font.family": "sans-serif"}):
    fig_b, ax_b = plt.subplots(figsize=(8, 4.8))

    for label, sub_df in [("Layer VQE", layer_vqe), ("Base VQE", base_vqe)]:
        color = COLORS[label]
        x = sub_df["n_layers"].values
        y = sub_df["mean"].values
        e = sub_df["sem"].values

        ax_b.errorbar(
            x, y, yerr=e,
            fmt="o-", color=color, linewidth=3, markersize=9,
            capsize=5, capthick=2, elinewidth=2,
            label=label, zorder=3,
        )
        ax_b.fill_between(x, y - e, y + e, alpha=0.15, color=color, zorder=2)

    ax_b.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.6, label="BKV (SW)")

    # Double-headed arrow annotating the gap at layer 2
    x_ann = 2
    lvqe_row = layer_vqe[layer_vqe["n_layers"] == x_ann]
    bvqe_row = base_vqe[base_vqe["n_layers"] == x_ann]
    if not lvqe_row.empty and not bvqe_row.empty:
        lvqe_y = lvqe_row["mean"].values[0]
        bvqe_y = bvqe_row["mean"].values[0]
        gap = lvqe_y - bvqe_y
        ax_b.annotate(
            "",
            xy=(x_ann + 0.13, bvqe_y),
            xytext=(x_ann + 0.13, lvqe_y),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.5),
            zorder=5,
        )
        ax_b.text(
            x_ann + 0.20, (lvqe_y + bvqe_y) / 2,
            f"$\\Delta = {gap:.3f}$",
            fontsize=11, color="#555555", va="center",
        )

    ax_b.set_xlabel("Number of layers", fontsize=15)
    ax_b.set_ylabel("Final approximation ratio", fontsize=15)
    ax_b.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_b.set_ylim(bottom=0.9, top=1.01)
    ax_b.tick_params(axis="both", labelsize=12)
    ax_b.legend(fontsize=12, loc="lower right")
    ax_b.grid(True, linestyle="--", alpha=0.35)
    fig_b.tight_layout()

    beamer_path = os.path.join(os.path.dirname(__file__), "results/model_comparison_beamer.pdf")
    fig_b.savefig(beamer_path, bbox_inches="tight")
    print(f"Saved → {beamer_path}")
