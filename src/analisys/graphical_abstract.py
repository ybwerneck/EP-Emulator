import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import os

# ── colour scheme ─────────────────────────────────────────────────────────────
NN_COLOR  = "#2166AC"   # dark blue
GP_COLOR  = "#1A7C3E"   # forest green
PCE_COLOR = "#D6604D"   # warm red

def family_color(model_name: str) -> str:
    m = model_name.upper()
    if m.startswith("NN"):  return NN_COLOR
    if m.startswith("GP"):  return GP_COLOR
    return PCE_COLOR

def family_label(model_name: str) -> str:
    m = model_name.upper()
    if m.startswith("NN"):  return "Neural Network"
    if m.startswith("GP"):  return "Gaussian Process"
    return "Polynomial Chaos"

SIZE_MAP = {100: 28, 200: 55, 500: 100, 1000: 165}

# ── true model per-sample cost (s) — derived from 5 K performance.csv ─────────
TRUE_PER_SAMPLE = {
    "A": 1538.4748  / 5000,   # EP model A,  n=3  params
    "B":  474.8863  / 5000,   # EP model B,  n=12 params
    "C": 144899.944 / 5000,   # Mechanical,  n=4  params
    "D": 148698.035 / 5000,   # Mechanical,  n=8  params
}

PROBLEM_LABELS = {
    "A": "EP-AP\n(n=3)",
    "B": "EP-AP\n(n=12)",
    "C": "Mechanical\n(n=4)",
    "D": "Mechanical\n(n=8)",
}
PROBS = ["A", "B", "C", "D"]

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "Results")

# ── load & enrich data ─────────────────────────────────────────────────────────
frames = []
for prob in PROBS:
    df = pd.read_csv(os.path.join(ROOT, f"inference_{prob}.csv"))
    if "Inference Time (100k samples) (s)" in df.columns:
        df = df.rename(columns={"Inference Time (100k samples) (s)": "InfTime_100k"})
    else:
        df = df.rename(columns={"Inference Time (s)": "InfTime_100k"})
    df["Speedup"]    = (TRUE_PER_SAMPLE[prob] * 1e5) / df["InfTime_100k"]
    df["Problem"]    = prob
    df["Family"]     = df["Model"].apply(family_label)
    df["Color"]      = df["Model"].apply(family_color)
    df["MarkerSize"] = df["Training Samples"].map(SIZE_MAP)
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# ── jitter helper (reproducible) ───────────────────────────────────────────────
def jittered_x(prob_idx: int, n: int, width: float = 0.22) -> np.ndarray:
    rng = np.random.default_rng(7 + prob_idx * 31)
    return prob_idx + rng.uniform(-width, width, n)

# ── layout ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5.2),
    gridspec_kw={"wspace": 0.22},
)

x_positions = np.arange(len(PROBS))

for ax, metric, ylabel in [
    (axes[0], "MARE",    "MARE"),
    (axes[1], "Speedup", "Speed-up"),
]:
    # ── light band to distinguish EP vs Mechanical ─────────────────────────────
    ax.axvspan(-0.55, 1.55, color="#E8F4FD", alpha=0.55, zorder=0)  # EP region
    ax.axvspan( 1.55, 3.55, color="#F0FAF2", alpha=0.55, zorder=0)  # Mech region

    # ── scatter each group ─────────────────────────────────────────────────────
    seen: set[str] = set()
    for i, prob in enumerate(PROBS):
        sub = data[data["Problem"] == prob].reset_index(drop=True)
        xs  = jittered_x(i, len(sub))
        for local_j, row in sub.iterrows():
            fam   = row["Family"]
            label = fam if fam not in seen else "_nolegend_"
            seen.add(fam)
            ax.scatter(
                xs[local_j],
                row[metric],
                color=row["Color"],
                s=row["MarkerSize"],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.5,
                label=label,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([PROBLEM_LABELS[p] for p in PROBS], fontsize=11.5, linespacing=1.4)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", which="minor", linestyle=":",  linewidth=0.3, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, len(PROBS) - 0.4)

    # ── region labels inside each panel ───────────────────────────────────────
    ax.text(0.5,  1.0, "Electrophysiology", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color="#1A6FA0",
            style="italic", clip_on=False)
    ax.text(2.5,  1.0, "Mechanical", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color="#1A6B2E",
            style="italic", clip_on=False)

# ── family legend (shared, bottom centre) ─────────────────────────────────────
family_patches = [
    mpatches.Patch(color=NN_COLOR,  label="Neural Network"),
    mpatches.Patch(color=GP_COLOR,  label="Gaussian Process"),
    mpatches.Patch(color=PCE_COLOR, label="Polynomial Chaos"),
]
fig.legend(
    handles=family_patches,
    fontsize=9.5,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.11),
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
)

# ── training-size legend (inside speed-up panel, top-left) ────────────────────
size_handles = [
    plt.scatter([], [], s=SIZE_MAP[n], color="grey", alpha=0.75,
                edgecolors="white", linewidths=0.5, label=f"N = {n}")
    for n in [100, 200, 500, 1000]
]
axes[1].legend(
    handles=size_handles,
    title="Training size",
    title_fontsize=9,
    fontsize=8.5,
    loc="upper left",
    frameon=True,
    framealpha=0.92,
    edgecolor="#cccccc",
)

fig.suptitle(
    "Surrogate Modeling in Cardiac Mechanics and Electrophysiology",
    fontsize=14, fontweight="bold", y=1.02,
)

out = os.path.join(ROOT, "plots", "graphical_abstract.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
