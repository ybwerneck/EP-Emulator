import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

# -------------------------------
# Problems
probs = ["A", "B"]
#probs=["C", "D"]

# Custom problem names
problem_names = {
    "A": "Problem A_E",
    "B": "Problem B_E",
    "C": "Problem A_M",
    "D": "Problem B_M"
}

# QoI columns in CSV
qoi_cols = [
    "MARE_QoI_0",
    "MARE_QoI_1",
    "MARE_QoI_2",
    "MARE_QoI_3",
    "MARE_QoI_4",
    "MARE_QoI_5"
]

# Custom QoI labels
qoi_labels = {
    "MARE_QoI_0": "V_rest",
    "MARE_QoI_1": "V_peak",
    "MARE_QoI_2": "dVdt_max",
    "MARE_QoI_3": "APD80",
    "MARE_QoI_4": "APD50",
    "MARE_QoI_5": "APD30"
}

if(probs==["C", "D"]):
    qoi_labels = {
    "MARE_QoI_0": "alfa1",
    "MARE_QoI_1": "beta1",
    "MARE_QoI_2": "alfa2",
    "MARE_QoI_3": "beta2",
    "MARE_QoI_4": "Volume",
    "MARE_QoI_5": "fibrestretch"
}


# Model display names
display_names = [
    "NN_S", "NN_M", "NN_G",
    "GP_S", "GP_M", "GP_G",
    "PC_2", "PC_3", "PC_5"
]

# -------------------------------
# Load data
data_dict = {
    prob: pd.read_csv(f"Results/inference_{prob}.csv")
    for prob in probs
}

def convert_set_size(set_str):
    return float(set_str)

# -------------------------------
# Plot heatmaps
for prob in probs:

    df = data_dict[prob]

    # Map model names
    model_to_display = dict(zip(df["Model"].unique(), display_names))
    df["Display Model"] = df["Model"].map(model_to_display)

    # Convert training size
    df["Training Set Size"] = df["Training Samples"].apply(convert_set_size)

    # Create grouped model label
    df["Model_Label"] = (
        df["Display Model"]
        + "_"
        + df["Training Set Size"].astype(int).astype(str)
    )

    # Keep models grouped
    df = df.sort_values(["Display Model", "Training Set Size"])

    # Build heatmap matrix
    heatmap_df = df.set_index("Model_Label")[qoi_cols]

    # Replace column names with custom QoI labels
    heatmap_df = heatmap_df.rename(columns=qoi_labels)

    # -------------------------------
    plt.figure(figsize=(14, 12))

    ax = sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        norm=colors.LogNorm(vmin=1e-5, vmax=1),
        linewidths=0.7,
        linecolor="black",
        annot=True,
        fmt=".1e",
        annot_kws={"size":14},
        cbar_kws={
            "label": "MARE",
            "shrink": 1.0,   # increase height (try 1.2–1.4 if needed)
            "aspect": 15     # thickness of colorbar (smaller = thicker)
        }
    )

    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)   # tick font size
    cbar.set_label("MARE", fontsize=22) # label font
    plt.title(f"{problem_names[prob]} – QoI Error Heatmap", fontsize=28)

    plt.xlabel("QoI", fontsize=24)
    plt.ylabel("Model / Training Size", fontsize=24)

    plt.xticks(rotation=30, fontsize=18)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    plt.savefig(f"Results/heatmap_qoi_{prob}.png", dpi=300)
    print("f")
    plt.close()

print("Heatmaps generated.")