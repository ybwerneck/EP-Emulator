import numpy as np
import pandas as pd
import os, sys, pickle, time, json
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ============================================================
# Gradient refinement
# ============================================================

def gradient_refine(emulator, P_init, Y_target, dist=None,
                    lr=0.02, num_steps=20, lambda_prior=0.0,
                    P_min=0.75, P_max=1.25, device="cuda"):
    """
    Quick gradient refinement for a candidate P_init.

    Returns:
        refined_P: numpy array
        best_loss: float
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    device = torch.device(device)
    P = torch.tensor(P_init[None, :], dtype=torch.float32, device=device, requires_grad=True)
    Y_t = torch.tensor(Y_target[None, :], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([P], lr=lr)

    best_loss = float('inf')
    best_P = P.clone().detach()

    for _ in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        Q_pred = emulator.forward(P)

        # Smooth L1 loss for better stability with few steps
        loss_data = F.smooth_l1_loss(Q_pred, Y_t)

        # Optional prior
        loss_prior = 0.0
        if dist is not None and lambda_prior > 0:
            pdf_vals = dist.pdf(P.detach().cpu().numpy()) + 1e-12
            loss_prior = -np.log(pdf_vals).mean()
            loss_prior = torch.tensor(loss_prior, dtype=torch.float32, device=device)

        loss = loss_data + lambda_prior * loss_prior
        loss.backward()
        optimizer.step()

        # Clamp parameters
        with torch.no_grad():
            P[:] = P.clamp(P_min, P_max)

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_P = P.clone().detach()

    return best_P.cpu().numpy().flatten(), best_loss


# ============================================================
# Utilities
# ============================================================

def run_emulator(P, emulator):
    """Evaluate emulator on parameter matrix P (N x n_params)."""
    return emulator.predict(P)


def prior_loss(P, dist):
    pdf_vals = dist.pdf(P.T) + 1e-12
    return -np.log(pdf_vals)


def pick_three(pop_size):
    r1, r2, r3 = np.zeros(pop_size,int), np.zeros(pop_size,int), np.zeros(pop_size,int)
    for j in range(pop_size):
        choices = np.arange(pop_size)
        choices = choices[choices != j]
        sel = np.random.choice(choices, 3, replace=False)
        r1[j], r2[j], r3[j] = sel
    return r1, r2, r3


def compute_param_error(P, X_true):
    """
    P: (batch, pop, n_params)
    X_true: (batch, n_params)
    returns: (batch, pop, n_params)
    """
    return np.abs(P - X_true[:, None, :])


def compute_y_error(Y_pred, Y_true):
    """
    Y_pred: (batch, pop, n_outputs)
    Y_true: (batch, n_outputs)
    returns: (batch, pop, n_outputs)
    """
    return np.abs(Y_pred - Y_true[:, None, :]) / (np.abs(Y_true[:, None, :]) + 1e-12)


# ============================================================
# Inverse problem with Differential Evolution
# ============================================================

def inverse_problem_DE(emulator, X, Y, dist,
                       batch_size=10, pop_size=100, num_iters=2000,
                       F=0.6, CR=0.7, prior_weight=0.0,
                       P_min=0.75, P_max=1.25,
                       results_dir="Results/InverseProblem",
                       checkpoint_interval=50,
                       grad_refine=False,
                       indices=None):

    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------------
    # Batch selection
    # --------------------------------------------------------
    indices = np.random.choice(len(X), batch_size, replace=False) if indices is None else indices
    X_true, Y_true = X[indices], Y[indices]

    n_params = X_true.shape[1]
    n_outputs = Y_true.shape[1]

    print("Selected batch indices:", indices)

    # --------------------------------------------------------
    # Population initialization
    # --------------------------------------------------------
    np.random.seed(42)

    P_candidates = np.array([
        dist.sample(pop_size, rule="latin_hypercube").T
        for _ in range(batch_size)
    ])  # (batch, pop, n_params)

    P_best = P_candidates.copy()
    best_loss = np.full((batch_size, pop_size), np.inf)

    # --------------------------------------------------------
    # Instrumentation storage (FULL)
    # --------------------------------------------------------
    param_error_hist = []   # (it, batch, pop, n_params)
    y_error_hist = []       # (it, batch, pop, n_outputs)
    loss_hist = []          # (it, batch, pop)

    history_best = []
    history_median = []
    iter_times = []

    t0 = time.time()

    # --------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------
    P_flat = P_candidates.reshape(batch_size * pop_size, n_params)
    Y_pred_all = run_emulator(P_flat, emulator)

    for i in range(batch_size):
        start, end = i * pop_size, (i+1) * pop_size
        preds = Y_pred_all[start:end]
        mse = np.mean((preds - Y_true[i])**2, axis=1)
        best_loss[i] = mse

    # --------------------------------------------------------
    # Evolution loop
    # --------------------------------------------------------
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
            mse = np.mean((preds - Y_true[i])**2, axis=1)

            prior_reg = np.abs(prior_weight * prior_loss(P_trial[i], dist))
            total = mse + prior_reg

            replace = total < best_loss[i]
            best_loss[i, replace] = total[replace]
            P_best[i, replace] = P_trial[i, replace]

        P_candidates = P_best.copy()

   
        # ----------------------------------------------------
        # Gradient refinement (optional)
        # ----------------------------------------------------
        if grad_refine and (it+1) % 1 == 0:
            for i in range(batch_size):
                j_best = np.argmin(best_loss[i])
                P_start = P_best[i, j_best].copy()
                Y_t = Y_true[i]

                P_refined, loss_ref = gradient_refine(
                    emulator, P_start, Y_t, dist,
                    lr=1e-4, num_steps=5, lambda_prior=0,
                    P_min=P_min, P_max=P_max
                )

                print(np.mean(loss_ref<best_loss[i, j_best]), f"Refinement improved loss: {best_loss[i, j_best]:.6e} -> {loss_ref:.6e}")
                if loss_ref < best_loss[i, j_best]:
                    P_best[i, j_best] = P_refined
                    best_loss[i, j_best] = loss_ref
            print(f"Iter {it+1}/{num_iters} after refinement | Best loss {best_loss.min():.6e}")
        # ----------------------------------------------------
        # Checkpoint logging
        # ----------------------------------------------------
        if (it+1) % checkpoint_interval == 0 or (it+1) == num_iters:

                    # ----------------------------------------------------
            # Full instrumentation
            # ----------------------------------------------------
            # Parameter errors
            param_err = compute_param_error(P_best, X_true)
            param_error_hist.append(param_err.copy())

            # Y errors
            P_flat = P_best.reshape(batch_size * pop_size, n_params)
            Y_pred_all = run_emulator(P_flat, emulator)
            Y_pred_all = Y_pred_all.reshape(batch_size, pop_size, n_outputs)

            y_err = compute_y_error(Y_pred_all, Y_true)
            y_error_hist.append(y_err.copy())

            # Loss
            loss_hist.append(best_loss.copy())

            # Scalars
            history_best.append(best_loss.min())
            history_median.append(np.median(best_loss))

            iter_times.append(time.time() - t_iter)

            mean_time = np.mean(iter_times[-checkpoint_interval:])
            print(f"Iter {it+1}/{num_iters} | Best loss {history_best[-1]:.6e} | mean iter time {mean_time:.3f}s")

            # Convergence plot
            plt.figure()
            plt.plot(history_best, label="Best")
            plt.plot(history_median, label="Median")
            plt.yscale("log")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Convergence")
            plt.savefig(os.path.join(results_dir, "convergence.png"), dpi=200)
            plt.close()

            # ========================================================
            # Save full scientific instrumentation
            # ========================================================

            np.save(os.path.join(results_dir, "param_error_full.npy"),
                    np.array(param_error_hist))
            #print(param_error_hist[-1])  # shape: (batch, pop, n_params)
            # shape: (iters, batch, pop, n_params)

            np.save(os.path.join(results_dir, "y_error_full.npy"),
                    np.array(y_error_hist))
            # shape: (iters, batch, pop, n_outputs)

            np.save(os.path.join(results_dir, "loss_full.npy"),
                    np.array(loss_hist))
            # shape: (iters, batch, pop)

            np.save(os.path.join(results_dir, "P_best_final.npy"), P_best)
            np.save(os.path.join(results_dir, "X_true.npy"), X_true)
            np.save(os.path.join(results_dir, "Y_true.npy"), Y_true)
            np.save(os.path.join(results_dir, "indices.npy"), indices)

            summary = {
                "total_runtime_sec": float(time.time() - t0),
                "final_best_loss": float(history_best[-1]),
                "final_median_loss": float(history_median[-1]),
                "mean_iter_time": float(np.mean(iter_times)),
                "batch_indices": indices.tolist(),
                "n_params": int(n_params),
                "n_outputs": int(n_outputs),
                "batch_size": int(batch_size),
                "pop_size": int(pop_size),
                "num_iters": int(num_iters)
            }

            with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    return P_best, history_best, summary
