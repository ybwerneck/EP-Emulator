import pandas as pd
import matplotlib.pyplot as plt

# Define model names
names = ["NN-s", "NN-m", "NN-l", "GP-skt", "GP-torc", "PCE2", "PCE3", "PCE5"]

# Create a dictionary to store data for combined plot
combined_data = {}

for P in ["A", "B"]:
    # Read the results CSV
    results_df = pd.read_csv(f'Results/prob_{P}/validation_results.csv')
    combined_data[P] = results_df

    # Individual plots
    plt.figure(figsize=(6, 6))

    # Scatter plot for each problem
    plt.scatter(results_df["MSE"], results_df["Inference Time (s)"], s=20, c='blue', alpha=0.7, edgecolors='black')
    plt.title(f"Model Performance: Elapsed Time vs Accuracy (Problem {P})", fontsize=16)
    plt.xlabel("Inference Time (s)", fontsize=14)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=14)

    # Add annotations for models
    for i, row in results_df.iterrows():
        plt.text(row["MSE"], row["Inference Time (s)"], f'{names[i]}', fontsize=10, ha='right')

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e-3, 1e1)

    # Save individual plot
    plt.savefig(f"Results/prob_{P}/inf_results.png")
    plt.close()

# Combined plot with one subplot per folder
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

for idx, P in enumerate(["A", "B", "C"]):
    results_df = combined_data[P]
    axs[idx].scatter(results_df["MSE"], results_df["Inference Time (s)"], s=150, c='blue', alpha=0.7, edgecolors='black')
    axs[idx].set_title(f"Problem {P}", fontsize=14)
    axs[idx].set_xlabel("Inference Time (s)", fontsize=12)
    if idx == 0:
        axs[idx].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)

    # Add annotations for models
    for i, row in results_df.iterrows():
        axs[idx].text(row["MSE"], row["Inference Time (s)"], f'{names[i]}', fontsize=15, ha='right')

    axs[idx].grid(alpha=0.3)
    axs[idx].set_yscale("log")
    axs[idx].set_xscale("log")
    axs[idx].set_xlim(1e-2, 3.3e3)
    axs[idx].set_ylim(1e-3, 1e2)

# Adjust layout and save combined plot
plt.tight_layout()
plt.savefig("Results/combined_inf_results.png")
plt.show()
