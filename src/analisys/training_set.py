import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Configuration
# -----------------------------
blue_colors = sns.color_palette("Blues", 6)[3:]
green_colors = sns.color_palette("Greens", 3)[:]
reds_colors  = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors 

# Internal codes -> LaTeX display names
prob_labels = {
    "A": r"$A_M$",
    "B": r"$B_M$",
    "C": r"$A_E$",
    "D": r"$B_E$"
}

probs = ["A","B","C","D"]
data_dict = {p: pd.read_csv(f"Results/inference_{p}.csv") for p in probs}

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

# -----------------------------
# Prepare learning-curve data
# -----------------------------
records = []

for prob in probs:
    data = data_dict[prob].copy()
    data['MARE_overall'] = data["MARE"]

    for model in model_names:
        sub = data[data['Model'] == model]

        for N in sorted(sub['Training Samples'].unique()):
            vals = sub[sub['Training Samples'] == N]['MARE_overall'].values

            records.append({
                "Problem": prob,
                "Model": model,
                "Training Samples": N,
                "Mean": vals.mean(),
                "Std": vals.std()
            })

lc_df = pd.DataFrame(records)

# -----------------------------
# Plot: one subplot per problem
# -----------------------------
n_probs = len(probs)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=False)
axes = axes.flatten()
if n_probs == 1:
    axes = [axes]

for ax, prob in zip(axes, probs):
    pdf = lc_df[lc_df['Problem'] == prob]

    for model in model_names:
        mdf = pdf[pdf['Model'] == model].sort_values("Training Samples")

        if len(mdf) == 0:
            continue

        N    = mdf['Training Samples'].values
        mean = mdf['Mean'].values

        ax.plot(
            N, mean,
            color=color_mapping[model],
            linewidth=2
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(prob_labels[prob], fontsize=15)
    ax.set_xlabel("Training set size $N$", fontsize=13)

# Shared y-label
axes[0].set_ylabel("Overall MARE", fontsize=13)

# -----------------------------
# Global legend
# -----------------------------
legend_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]

fig.legend(
    handles=legend_handles,
    title="Surrogate model",
    fontsize=11,
    title_fontsize=12,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5)
)

# -----------------------------
# Titles and layout
# -----------------------------
fig.suptitle("Learning curves across benchmark problems", fontsize=16)

plt.tight_layout()
#plt.subplots_adjust(right=0.82, top=0.90)
plt.savefig("Results/plots/learning_curve_mare_multiplot.png", dpi=600, bbox_inches="tight")
