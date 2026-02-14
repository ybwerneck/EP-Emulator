import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np
# -----------------------------
# Color palette for PCE models
# -----------------------------
pce_models = ["PCE_2", "PCE_3", "PCE_5"]
colors = sns.color_palette("Reds", len(pce_models))
color_mapping = {m: c for m, c in zip(pce_models, colors)}
marker_mapping = {'basis':'o', 'mc':'X'}

probs = ["A","B"]

# -----------------------------
# Load data
# -----------------------------
data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}
uqsa_dict = {}
for prob in probs:
    uqsa_df = pd.read_csv(f'Results/uq_sa_{prob}_pce.csv')

    def map_model_name(name):
        if "pce_model2" in name: return "PCE_2"
        if "pce_model3" in name: return "PCE_3"
        if "pce_model5" in name: return "PCE_5"
        return name

    uqsa_df['Model'] = uqsa_df['model'].apply(map_model_name)

    def get_subtype(name):
        if '_basis' in name: return 'basis'
        if '_mc' in name: return 'mc'
        return 'default'

    uqsa_df['subtype'] = uqsa_df['model'].apply(get_subtype)
    uqsa_dict[prob] = uqsa_df[['Model','subtype','mean_rel','S1_rel','uq_time_s','sa_time_s','model']]

metrics = [
    ("S1_rel",   "SA Error",   "log"),
    ("mean_rel", "UQ Error",   "log"),
    ("sa_time_s","SA Cost [s]","log"),
    ("uq_time_s","UQ Cost [s]","log"),
]

metrics = [
    ("S1_rel",   "SA Error"),
    ("mean_rel", "UQ Error"),
    ("sa_time_s","SA Cost [s]"),
    ("uq_time_s","UQ Cost [s]"),
]

metrics = [
    ("S1_rel",   "SA Error"),
    ("mean_rel", "UQ Error"),
    ("sa_time_s","SA Cost [s]"),
    ("uq_time_s","UQ Cost [s]"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

bar_width = 0.6
group_gap = 1.5   # gap between Problem A and B
inner_gap = 0.8   # gap between basis and MC

for idx, (col_name, ylabel) in enumerate(metrics):

    ax = axes[idx]

    x_positions = []
    labels = []
    means = []
    mins = []
    maxs = []

    x = 0

    for prob in probs:  # ["A","B"]

        dfp = uqsa_dict[prob]
        dfp = dfp[dfp["Model"] == "PCE_2"]

        for method in ["basis", "mc"]:

            sub = dfp[dfp["subtype"] == method][col_name].dropna()

            if len(sub) == 0:
                mean = np.nan
                vmin = np.nan
                vmax = np.nan
            else:
                mean = sub.mean()
                vmin = sub.min()
                vmax = sub.max()

            x_positions.append(x)
            labels.append(f"{prob}\n{method}")
            means.append(mean)
            mins.append(vmin)
            maxs.append(vmax)

            x += inner_gap

        x += group_gap  # space between problems

    # bars
    ax.bar(
        x_positions,
        means,
        width=bar_width,
        edgecolor="black",
        alpha=0.75
    )

    # min–max lines
    for xi, vmin, vmax in zip(x_positions, mins, maxs):
        ax.vlines(xi, vmin, vmax, linewidth=2)

    # mean markers
    #ax.scatter(x_positions, means, s=40, zorder=3)

    # formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    #ax.grid(True, axis="y", alpha=0.3)

    # log scale for errors and costs
    ax.set_yscale("log")

fig.suptitle("PCE(2) UQ/SA Summary — Problems A and B", fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Results/plots/pce2_4panel_summary.png", dpi=600, bbox_inches="tight")
plt.close(fig)

