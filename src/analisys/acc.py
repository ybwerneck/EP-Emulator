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
reds_colors = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors 

# File keys (used for CSV loading)
probs = ["A", "B", "C", "D"]

# LaTeX display labels
latex_labels = {
    "A": r"$A_E$",
    "B": r"$B_E$",
    "C": r"$A_M$",
    "D": r"$B_M$"
}

data_dict = {p: pd.read_csv(f"Results/inference_{p}.csv") for p in probs}

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

# -----------------------------
# Prepare data
# -----------------------------
mean_values = []
std_values = []
model_labels = []
prob_labels = []

for prob in probs:
    data = data_dict[prob].copy()
    data['MARE_overall'] = data["MARE"].values
    
    for model in model_names:
        ys = data.loc[data['Model'] == model, 'MARE_overall'].values
        mean_values.append(ys.mean())
        std_values.append(ys.std())
        model_labels.append(model)
        prob_labels.append(prob)

plot_df = pd.DataFrame({
    'Problem': prob_labels,
    'Model': model_labels,
    'Mean': mean_values,
    'Std': std_values
})

# -----------------------------
# Plotting layout
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharey=False)

groups = [
    ["A", "B"],   # top subplot  -> A_E, B_E
    ["C", "D"]    # bottom subplot -> A_M, B_M
]

total_width = 0.8
n_models = len(model_names)
bar_width = total_width / n_models

for ax, group_probs in zip(axes, groups):
    x = np.arange(len(group_probs))

    for i, model in enumerate(model_names):
        model_data = plot_df[
            (plot_df['Model'] == model) &
            (plot_df['Problem'].isin(group_probs))
        ].sort_values("Problem")

        ax.bar(
            x + (i - n_models/2) * bar_width + bar_width/2,
            model_data['Mean'],
            yerr=model_data['Std'],
            width=bar_width * 0.9,
            color=color_mapping[model],
            edgecolor='black',
            capsize=4
        )

    ax.set_xticks(x)
    ax.set_xticklabels([latex_labels[p] for p in group_probs], fontsize=13)
    ax.set_yscale("log")
    ax.set_ylabel("Overall MARE", fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)

# -----------------------------
# Legend (global)
# -----------------------------
legend_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]
fig.legend(
    handles=legend_handles,
    title="Emulator",
    fontsize=12,
    title_fontsize=12,
    loc='center right',
    bbox_to_anchor=(0.94, 0.5)
)


fig.supxlabel("Problem", fontsize=14)
fig.suptitle("Surrogate validation accuracy across problems", fontsize=17)

plt.tight_layout()
plt.subplots_adjust(right=0.82, top=0.92)
plt.savefig("Results/plots/mare_overall_surrogates_AB_CD.png", dpi=600, bbox_inches="tight")
