import numpy as np
import pandas as pd
import pickle
import time  # Import the time module



import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


import sys
sys.path.append("/home/yan/EP-Emulator/src/")



def evaluate_model(model, x_val, y_val, num_runs=10, n_infer_samples=100_000):
    """
    Evaluate a single model on validation data, compute metrics,
    and measure inference time using 100k random new samples.

    Returns a dictionary with metrics, inference times, and metadata.
    """

    print(f"Evaluating model: {getattr(model,'name','Unknown')}")
    print("Validation Shapes:", np.shape(x_val), np.shape(y_val))

    metrics = {
        "MARE": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred) / np.abs(y_true)),
        "MSE": lambda y_true, y_pred: np.mean(((y_true - y_pred) / np.abs(y_true)) ** 2),
        "R2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }

    num_qois = y_val.shape[1]
    for i in range(num_qois):
        metrics[f"MARE_QoI_{i}"] = lambda y_true, y_pred, i=i: np.mean(
            np.abs(y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i])
        )
        metrics[f"MSE_QoI_{i}"] = lambda y_true, y_pred, i=i: np.mean(
            ((y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i])) ** 2
        )

    results = {}

    # -------------------------------------------------
    # Inference benchmark on 100k NEW random samples
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Metrics computed on validation set (not random set)
    # -------------------------------------------------
    y_pred_val = model.predict(x_val, meas_time=False)
    print("Prediction Shapes:", np.shape(y_pred_val))

    for metric_name, metric_fn in metrics.items():
        results[metric_name] = metric_fn(y_val, y_pred_val)

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------
    results["Training Time (s)"] = getattr(model, "metadata", {}).get("time_train", "-1")
    results["Memory (gpu)"] = getattr(model, "metadata", {}).get("gpu_memory_MB", "-1")
    results["Training Iterations"] = getattr(model, "metadata", {}).get("training_epochs", "-1")

    print(results)
    return results


def evaluate_models(models, x_val, y_val,  num_runs=10):
    """
    Evaluate multiple models on the same validation data.

    Args:
        models: list of model objects with .predict method
        x_val: np.ndarray
        y_val: np.ndarray
        subset_sizes: list of int, optional
        num_runs: int, number of runs for averaging inference time

    Returns:
        pd.DataFrame: results for all models
    """
    all_results = []
    for i, model in enumerate(models):
        print(f"Evaluating model {i+1}/{len(models)}: {getattr(model,'name','Unknown')}")
        print("a")
        res = evaluate_model(model, x_val, y_val, num_runs=num_runs)
        print("b")
        res["id"] = i
        all_results.append(res)

    return pd.DataFrame(all_results)


import os
import pickle
import pandas as pd
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.surrogate_models.DD_Models import ModelInterface as Surrogate
from src.surrogate_models.gaussian_process import *

def evaluate_model_folder(models_folder, x_val, y_val, output_csv, subset_sizes=None, num_runs=10):
    """
    Load all models from a folder, evaluate them, and save results to CSV.

    Args:
        models_folder: str, path to folder containing model files (pickle .pkl or .pth)
        x_val: np.ndarray, validation inputs
        y_val: np.ndarray, validation outputs
        output_csv: str, path to write the results CSV
        subset_sizes: list of int, optional, subset sizes for inference time evaluation
        num_runs: int, number of runs to average inference time
    """
    # Collect all model files
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

    # Evaluate all models using the previously defined function
    results_df = evaluate_models(models, x_val, y_val, num_runs=num_runs)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return results_df




import argparse
import pandas as pd

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
                        help="Number of runs to average inference time. Default: 5")
    
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
        num_runs=args.num_runs
    )

    print("Top results:")
    print(results_df.head())

if __name__ == "__main__":
    main()
