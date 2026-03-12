import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------
# Style (publication friendly)
# ------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# ------------------------------------------------
# Data paths
# ------------------------------------------------
base = "data/Generated_Data_5K"

models = {
    "A_E": "ModelA",
    "B_E": "ModelB",
    "A_M": "ModelC",
    "B_M": "ModelD"
}

# ------------------------------------------------
# Load datasets
# ------------------------------------------------
data = {}
max_qois = 0

for label, folder in models.items():
    path = os.path.join(base, folder, "Y.csv")
    df = pd.read_csv(path)
    data[label] = df
    max_qois = max(max_qois, df.shape[1])

# ------------------------------------------------
# Figure
# ------------------------------------------------
fig, axes = plt.subplots(
    len(models),
    max_qois,
    figsize=(4*max_qois, 3.5*len(models))
)

# ------------------------------------------------
# Plot
# ------------------------------------------------
for row, (label, df) in enumerate(data.items()):

    cols = df.columns

    for col_idx in range(max_qois):

        ax = axes[row, col_idx]

        if col_idx < len(cols):

            qoi = cols[col_idx]

            sns.histplot(
                df[qoi],
                bins=40,
            #    stat="density",
                kde=False,
                color="#4C72B0",
                edgecolor="black",
                ax=ax
            )

            ax.set_title(qoi)

        else:
            ax.axis("off")

        if col_idx == 0:
            ax.set_ylabel(label, fontsize=16, rotation=90)

plt.tight_layout()

# ------------------------------------------------
# Save
# ------------------------------------------------
os.makedirs("Results/plots", exist_ok=True)

plt.savefig(
    "Results/plots/QoI_histograms_all_models.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: Results/plots/QoI_histograms_all_models.png")