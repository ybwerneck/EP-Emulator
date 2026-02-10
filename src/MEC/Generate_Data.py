import os
import time
import csv
import sys
import numpy as np
import pandas as pd
import chaospy as cp

# Add path to your models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.MEC.ModelsCD import TisoModel, Ho8Model  # Adjust import if necessary

# ------------------------------------------------------------
# Sample size parsing
# ------------------------------------------------------------
def parse_sample_size(arg):
    try:
        value = float(arg)
        return int(value * 1000)
    except ValueError:
        raise ValueError("Invalid argument. Use e.g. 100 (for 100k) or 0.1 (for 100).")

sample_size = parse_sample_size(sys.argv[1]) if len(sys.argv) > 1 else 100_000
print(f"Sample size: {sample_size}")

# ------------------------------------------------------------
# Distribution bounds
# ------------------------------------------------------------
c, d = 0.7, 1.3  # Example low/high bounds for all models

# ------------------------------------------------------------
# Utility to extract outputs (if needed)
# ------------------------------------------------------------
def extract_qois(Y):
    return Y

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------
output_dir = f"Generated_Data_testsse_{sample_size//1000 if sample_size >= 1000 else sample_size/1000}K"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ------------------------------------------------------------
# Experiment loop
# ------------------------------------------------------------
MODELS = [
    {"name": "Tiso", "class": TisoModel},
    {"name": "Ho8",  "class": Ho8Model},
]

for model_cfg in MODELS:
    model_name = model_cfg["name"]
    model_class = model_cfg["class"]

    print(f"\nRunning model: {model_name}")

    # -----------------------------
    # Instantiate model
    # -----------------------------
    model = model_class()

    performance = []

    # -----------------------------
    # Sampling
    # -----------------------------
    start = time.time()
    dist = model.get_dist(low=c, high=d)
    print(dist)
    X = dist.sample(sample_size)
    end = time.time()
    sampling_time = end - start
    performance.append(["Sampling", f"{sampling_time:.4f}"])
    print(f"{model_name} - Sampling: {sampling_time:.4f} seconds")

    # -----------------------------
    # Run model
    # -----------------------------
    start = time.time()
    Y = model.run_model(X)  # Adjust nproc based on your system
    if Y is not None:
        Y = extract_qois(Y)
        end = time.time()
        run_time = end - start
        performance.append(["Running Model", f"{run_time:.4f}"])
        print(f"{model_name} - Running Model: {run_time:.4f} seconds")

        # -----------------------------
        # Save data
        # -----------------------------
        start = time.time()
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        pd.DataFrame(X.T, columns=[f"Input_{i+1}" for i in range(X.shape[0])]).to_csv(
            os.path.join(model_dir, "X.csv"), index=False
        )
        pd.DataFrame(Y.T, columns=[f"Output_{i+1}" for i in range(Y.T.shape[1])]).to_csv(
            os.path.join(model_dir, "Y.csv"), index=False
        )
        end = time.time()
        save_time = end - start
        performance.append(["Saving Data", f"{save_time:.4f}"])
        print(f"{model_name} - Saving Data: {save_time:.4f} seconds")

        # -----------------------------
        # Save performance CSV
        # -----------------------------
        perf_path = os.path.join(model_dir, "performance.csv")
        with open(perf_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Time (seconds)"])
            writer.writerows(performance)

print(f"\nAll results saved to: {output_dir}")
# ------------------------------------------------------------