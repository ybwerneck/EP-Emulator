import numpy as np
import pandas as pd
import pickle
import time
import os
import sys
from pathlib import Path
import argparse

sys.path.append("/home/yan/EP-Emulator/src/")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.surrogate_models.DD_Models import ModelInterface as Surrogate
from src.surrogate_models.gaussian_process import *

# ============================================================
# Core evaluation
# ============================================================

def evaluate_model(model, x_val, y_val, num_runs=10, n_infer_samples=int(1e5), ignore_qois=None):

    if ignore_qois is None:
        ignore_qois = []

    print(f"Evaluating model: {getattr(model,'name','Unknown')}")
    print("Validation Shapes:", np.shape(x_val), np.shape(y_val))
    print("Ignoring QoIs:", ignore_qois)

    # -----------------------------
    # Helpers
    # -----------------------------
    def filter_qois(y):
        if len(ignore_qois) == 0:
            return y
        mask = [i for i in range(y.shape[1]) if i not in ignore_qois]
        return y[:, mask]
    y_val  = filter_qois(y_val)
    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "MARE": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred) / np.abs(y_true)),
        "MSE":  lambda y_true, y_pred: np.mean(((y_true - y_pred) / np.abs(y_true)) ** 2),
        "R2":   lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }

    num_qois = y_val.shape[1] 
    for i in range(num_qois):
        if i in ignore_qois:
            continue
        metrics[f"MARE_QoI_{i}"] = lambda y_true, y_pred, i=i: np.mean(
            np.abs(y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i])
        )
        metrics[f"MSE_QoI_{i}"] = lambda y_true, y_pred, i=i: np.mean(
            ((y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i])) ** 2
        )

    results = {}

    # ============================================================
    # Inference benchmark (unchanged by QoI filtering)
    # ============================================================
    x_min = np.min(x_val, axis=0)
    x_max = np.max(x_val, axis=0)
    x_rand = np.random.uniform(low=x_min, high=x_max, size=(n_infer_samples, x_val.shape[1]))

    times = []
    for _ in range(num_runs):
        _, elapsed_time = model.predict(x_rand, meas_time=True)
        times.append(elapsed_time)

    avg_time = np.mean(times)

    results["Model"] = getattr(model, "name", "Unknown")
    results["Training Samples"] = getattr(model, "n_train_samples", "-1")
    results["Inference Time (100k samples) (s)"] = avg_time

    # ============================================================
    # Validation metrics
    # ============================================================
    y_pred_val = model.predict(x_val, meas_time=False)

    # Filter ignored QoIs
   
    y_pred_f = filter_qois(y_pred_val)

    print("Prediction Shapes (filtered):", np.shape(y_pred_f))

    for metric_name, metric_fn in metrics.items():
        results[metric_name] = metric_fn(y_val, y_pred_f)

    # ============================================================
    # Metadata
    # ============================================================
    results["Training Time (s)"] = getattr(model, "metadata", {}).get("time_train", "-1")
    results["Memory (gpu)"] = getattr(model, "metadata", {}).get("gpu_memory_MB", "-1")
    results["Training Iterations"] = getattr(model, "metadata", {}).get("training_epochs", "-1")

    print(results)
    return results


# ============================================================
# Multi-model evaluation
# ============================================================

def evaluate_models(models, x_val, y_val, num_runs=10, ignore_qois=None):

    all_results = []
    for i, model in enumerate(models):
        print(f"Evaluating model {i+1}/{len(models)}: {getattr(model,'name','Unknown')}")
        res = evaluate_model(
            model,
            x_val,
            y_val,
            num_runs=num_runs,
            ignore_qois=ignore_qois
        )
        res["id"] = i
        all_results.append(res)

    return pd.DataFrame(all_results)


# ============================================================
# Folder loader
# ============================================================

def evaluate_model_folder(models_folder, x_val, y_val, output_csv, num_runs=10, ignore_qois=None):

    model_files = [f for f in os.listdir(models_folder) if f.endswith(('.pkl','.pth'))]
    models = []

    for f in model_files:
        model_path = os.path.join(models_folder, f)
        try:
            model = Surrogate.load(model_path)
            models.append(model)
            print(f"Loaded model: {f}")
        except Exception as e:
            print(f"Failed to load model {f}: {e}")

    results_df = evaluate_models(
        models,
        x_val,
        y_val,
        num_runs=num_runs,
        ignore_qois=ignore_qois
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return results_df


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a folder of models on a validation dataset.")

    parser.add_argument("--models_folder", type=str, required=True,
                        help="Path to folder containing model files (.pkl or .pth).")
    parser.add_argument("--x_val", type=str, required=True,
                        help="Path to CSV file with validation inputs (X).")
    parser.add_argument("--y_val", type=str, required=True,
                        help="Path to CSV file with validation outputs (Y).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to CSV file where evaluation results will be saved.")

    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs to average inference time.")

    parser.add_argument("--ignore_qois", nargs="*", type=int, default=[],
                        help="QoI indices to ignore (0-based). Example: --ignore_qois 2 3")

    args = parser.parse_args()

    # Load validation data
    x_val = pd.read_csv(args.x_val).values
    y_val = pd.read_csv(args.y_val).values

    # Evaluate models
    results_df = evaluate_model_folder(
        models_folder=args.models_folder,
        x_val=x_val,
        y_val=y_val,
        output_csv=args.output_csv,
        num_runs=args.num_runs,
        ignore_qois=args.ignore_qois
    )

    print("Top results:")
    print(results_df.head())


if __name__ == "__main__":
    main()
