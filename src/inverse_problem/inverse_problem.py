import numpy as np
import pandas as pd
import os, sys, pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.surrogate_models.neural_networks import NModel
# from src.surrogate_models.gaussian_process import GPModel
# from src.surrogate_models.pce import PCEModel


def run_emulator(P, emulator):
    """Evaluate emulator or model on parameter matrix P (n_candidates x n_params)."""
    return emulator.predict(P)

def prior_loss(P, dist):
    """Negative log prior (regularization)."""
    pdf_vals = dist.pdf(P.T) + 1e-12
    return -np.log(pdf_vals)

def pick_three(pop_size):
    """Helper for DE mutation."""
    r1, r2, r3 = np.zeros(pop_size,int), np.zeros(pop_size,int), np.zeros(pop_size,int)
    for j in range(pop_size):
        choices = np.arange(pop_size)
        choices = choices[choices != j]
        sel = np.random.choice(choices, 3, replace=False)
        r1[j], r2[j], r3[j] = sel
    return r1, r2, r3

# -----------------------------------------------------------
# Inverse problem with DE
# -----------------------------------------------------------

import os, time, json
import numpy as np
import matplotlib.pyplot as plt

def inverse_problem_DE(emulator, X, Y, dist,
                       batch_size=10, pop_size=100, num_iters=2000,
                       F=0.8, CR=0.7, prior_weight=0.00,
                       P_min=0.75, P_max=1.25,
                       results_dir="Results/InverseProblem",
                       checkpoint_interval=5):
    """
    Perform inverse problem using Differential Evolution and a surrogate or true model.
    Saves instrumentation (loss curves, parity plots, validation, timings, parameter errors).
    """
    os.makedirs(results_dir, exist_ok=True)

    # Select ground truth batch
    indices = np.random.choice(len(X), batch_size, replace=False)
    X_true, Y_true = X[indices], Y[indices]
    n_params = X_true.shape[1]
    print("Selected batch:", indices)

    # Init population
    np.random.seed(42)
    P_candidates = np.array([dist.sample(pop_size, rule="latin_hypercube").T
                             for _ in range(batch_size)])
    P_best = P_candidates.copy()
    best_loss = np.full((batch_size, pop_size), np.inf)

    # History trackers
    history_best, history_median, history_val = [], [], []
    history_param_error = []  # Track parameter error
    iter_times = []
    t0 = time.time()

    # Initial evaluation
    P_flat = P_candidates.reshape(batch_size * pop_size, n_params)
    Y_pred_all = run_emulator(P_flat, emulator)

    for i in range(batch_size):
        start, end = i * pop_size, (i+1) * pop_size
        preds = Y_pred_all[start:end]
        invalid = ~np.isfinite(preds).all(axis=1)
        mse = np.mean((preds - Y_true[i])**2, axis=1)
      #  mse[invalid] = 1e6
        prior_reg = np.abs(prior_weight * prior_loss(P_candidates[i], dist))
        best_loss[i] = mse + prior_reg

    # Evolution loop
    for it in range(num_iters):
        t_iter = time.time()

        r1, r2, r3 = pick_three(pop_size)
        a, b, c = P_candidates[:, r1, :], P_candidates[:, r2, :], P_candidates[:, r3, :]
        mutants = a + F * (b - c)
        cross = np.random.rand(batch_size, pop_size, n_params) < CR
        P_trial = np.where(cross, mutants, P_candidates)
        P_trial = np.clip(P_trial, P_min, P_max)

        P_flat = P_trial.reshape(batch_size * pop_size, n_params)
        Y_pred_all = run_emulator(P_flat, emulator)

        for i in range(batch_size):
            start, end = i * pop_size, (i+1) * pop_size
            preds = Y_pred_all[start:end]
            invalid = ~np.isfinite(preds).all(axis=1)
            mse = np.mean((preds - Y_true[i])**2, axis=1)
        #    mse[invalid] = 1e6
            prior_reg = np.abs(prior_weight * prior_loss(P_trial[i], dist))
            total = mse + prior_reg

            replace = total < best_loss[i]
            best_loss[i, replace] = total[replace]
            P_best[i, replace] = P_trial[i, replace]

        P_candidates = P_best.copy()

        # Track losses
        history_best.append(best_loss.min())
        history_median.append(np.median(best_loss))

        # Track parameter error (mean absolute error per batch)
        param_errors = np.mean(np.abs(P_best - X_true[:, None, :]), axis=(1, 2))  # shape: (batch_size,)
        history_param_error.append(param_errors)

        # Iter time
        iter_times.append(time.time() - t_iter)

        # Checkpoint instrumentation
        if (it+1) % checkpoint_interval == 0 or (it+1) == num_iters:
            mean_time = np.mean(iter_times[-checkpoint_interval:])
            print(f"Iter {it+1}/{num_iters} | Best loss {history_best[-1]:.6f} "
                  f"| mean iter time {mean_time:.3f}s")

            # Convergence plot
            plt.figure()
            plt.plot(history_best, label="Best")
            plt.plot(history_median, label="Median")
            plt.yscale("log")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Convergence")
            plt.savefig(os.path.join(results_dir, f"convergence_iter.png"), dpi=200)
            plt.close()

            # Parameter error plot
            plt.figure()
            for i in range(batch_size):
                plt.plot([h[i] for h in history_param_error], label=f"Batch {i}")
            plt.xlabel("Iteration")
            plt.ylabel("Mean Parameter Error")
            plt.yscale("log")
            plt.title("Parameter Error Over Iterations")
           # plt.legend()
            plt.savefig(os.path.join(results_dir, "parameter_error_iter.png"), dpi=200)
            plt.close()

            # ---------------------------------------------------------
            # Parity plots (colored by Y error)
            # ---------------------------------------------------------
            P_final = P_best.copy()
            n_rows = int(np.ceil(n_params / 3))
            fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
            axes = axes.flatten()

            P_flat = P_final.reshape(batch_size * pop_size, n_params)
        #    Y_pred_all = run_emulator(P_flat, emulator)

            Y_errors = []
            for i in range(batch_size):
                start, end = i * pop_size, (i + 1) * pop_size
                true = Y_true[i]
                preds = Y_pred_all[start:end]
                rel_err = np.mean(np.abs(preds - true) / (np.abs(true) + 1e-12), axis=1)
                Y_errors.append(rel_err)
            Y_errors = np.concatenate(Y_errors)
        #    print(Y_pred_all[0])
#print(Y_errors[0])
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=0, vmax=1)
            # Cap errors at 1.0
            Y_errors_m = np.minimum(10*Y_errors, 1.0)
       #     print(Y_errors_m)
            for i in range(n_params):
                true_params = np.repeat(X_true[:, i], pop_size)
                recovered = P_final[:, :, i].flatten()

                sc = axes[i].scatter(
                    true_params, recovered,
                    c=Y_errors_m, cmap="viridis", vmin=0, vmax=1,   # enforce same 0–1 scale
                    alpha=0.7, s=25, edgecolors="none"
                )

                axes[i].plot([P_min, P_max], [P_min, P_max], 'r--')
                axes[i].set_xlabel(f"True Param {i+1}")
                axes[i].set_ylabel(f"Recovered Param {i+1}")
                axes[i].set_title(f"Parameter {i+1}")

            for j in range(n_params, len(axes)):
                fig.delaxes(axes[j])

            cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.95)
            cbar.set_label("Relative Y Error")
            plt.suptitle("Parity Plots for Parameters (Colored by Y Error)", fontsize=16)
            plt.savefig(os.path.join(results_dir, "emulator_inverse_parity.png"), dpi=300)
            plt.close(fig)

    # Metrics summary
    runtime = time.time() - t0
    summary = {
        "total_runtime_sec": runtime,
        "final_best_loss": float(history_best[-1]),
        "final_median_loss": float(history_median[-1]),
        "mean_iter_time": float(np.mean(iter_times)),
        "batch_indices": indices.tolist(),
        "final_param_error": [float(h[-1]) for h in history_param_error]
    }
    with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save final parameter error history
    np.save(os.path.join(results_dir, "history_param_error.npy"), np.array(history_param_error))

    return P_final, history_best, summary
