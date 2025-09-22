import torch
import numpy as np
import pickle
import pandas as pd
import time
import os, sys
import chaospy as cp
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.EP.ModelA import TTCellModelExt as modelA
from src.EP.ModelB import TTCellModelChannel as modelB
from src.EP.ModelC import TTCellModelFull as modelC
from src.surrogate_models.neural_networks import NModel
from true_model_inverse import run_model_to_array

# ---------------------------------------------------------------------
# Helper: differentiable prior penalty (same as your notebook)
# ---------------------------------------------------------------------
def prior_loss_from_cp_torch(P, dist, sigma=0.01):
    """
    Differentiable penalty for deviation from uniform (or normal) prior.
    P: (batch_size, n_params) torch tensor
    dist: chaospy distribution (list-like of marginals)
    """
    batch_size, n_params = P.shape
    penalty = torch.tensor(0.0, dtype=P.dtype, device=P.device)

    for i in range(n_params):
        d = dist[i]
        if isinstance(d, cp.Uniform):
            a = torch.tensor(d.lower, dtype=P.dtype, device=P.device)
            b = torch.tensor(d.upper, dtype=P.dtype, device=P.device)
            x = (P[:, i] - a) / (b - a)  # normalized to [0,1]
            # pairwise diffs
            diffs = x.unsqueeze(1) - x.unsqueeze(0)
            k = torch.exp(-0.5 * (diffs / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2.0 * np.pi, dtype=P.dtype, device=P.device)))
            density = k.mean(dim=1)
            uniform_density = torch.tensor(1.0, dtype=P.dtype, device=P.device)
            penalty += ((density - uniform_density) ** 2).mean()
        elif isinstance(d, cp.Normal):
            mu = torch.tensor(d.mu, dtype=P.dtype, device=P.device)
            sigma_n = torch.tensor(d.sigma, dtype=P.dtype, device=P.device)
            penalty += (0.5 * ((P[:, i] - mu) / sigma_n) ** 2 + torch.log(sigma_n * torch.sqrt(torch.tensor(2.0 * np.pi, dtype=P.dtype, device=P.device)))).mean()
        else:
            raise NotImplementedError(f"Distribution {type(d)} not implemented in prior_loss_from_cp_torch")
    return penalty

# ---------------------------------------------------------------------
# Main: Adam-based inverse function, same style/interface as DE version
# ---------------------------------------------------------------------
def inverse_problem_adam(
    model,           # emulator with .forward(torch.Tensor) -> torch.Tensor
    X,               # numpy array (N, d)
    Y,               # numpy array (N, q)
    dist,            # chaospy distribution list (marginals) or chaospy joint (indexable)
    batch_size=1024,
    num_iters=1000000,
    lr=5e-2,
    weight_decay=1e-4,
    lambda_prior=1e-4,
    P_min=0.75,
    P_max=1.25,
    device=None,
    checkpoint_interval=10000,
    results_dir="Results/InverseProblem",
    noise_every=10,
    noise_scale_factor=None,  # if None use lr each injection
    clip_grad_norm=5.0,
    stop_loss=1e-5,
    lr_stop_threshold=1e-7,
    verbose=True,
    true_model=None  # optional wrapper for validation (predict or run_model_to_array style)
):
    """
    Gradient-based inverse using AdamW. Mirrors the modular logic & style of your Adam script.
    Returns: P_opt (numpy), Q_opt (numpy), summary (dict)
    """
    os.makedirs(results_dir, exist_ok=True)

    # Device
    if device is None:
        device = getattr(model, "device", "cpu")
    torch_device = torch.device(device)

    # Select random batch
    np.random.seed(0)
    indices = np.random.choice(len(X), batch_size, replace=False)
    X_true = X[indices]
    Y_true = Y[indices]
    Q_target = torch.tensor(Y_true, dtype=torch.float32, device=torch_device)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        try:
            n_params = model.model.layers[0].in_features
        except Exception:
            n_params = X.shape[1]
    else:
        n_params = X.shape[1]

    if verbose:
        print("Selected batch indices:", indices)
        print("batch_size, n_params:", batch_size, n_params)
        print("Device used:", torch_device)

    # Initialize P as trainable tensor
    P = torch.randn((batch_size, n_params), requires_grad=True, device=torch_device)

    optimizer = torch.optim.AdamW([P], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10000, verbose=verbose)

    def loss_fn(pred, target):
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))

    # bookkeeping
    history_loss = []
    history_mare = []
    iter_times = []
    t_start = time.time()

    # ensure model in eval mode and no grad by default
    model.model.eval()

    # training loop
    for it in range(num_iters):
        t0 = time.time()
        optimizer.zero_grad()

        # forward
        Q_pred = model.forward(P)  # torch tensor (batch_size, q)

        mse_loss_val = loss_fn(Q_pred, Q_target)
        prior_loss_val = prior_loss_from_cp_torch(P, dist)
        loss = mse_loss_val + lambda_prior * prior_loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_([P], max_norm=clip_grad_norm)
        optimizer.step()

        # Langevin / noise injection
        if noise_every is not None and (it % noise_every == 0):
            noise_scale = optimizer.param_groups[0]['lr'] if noise_scale_factor is None else noise_scale_factor
            with torch.no_grad():
                P.add_(noise_scale * torch.randn_like(P))

        # clamp parameters
        with torch.no_grad():
            P.clamp_(P_min, P_max)

        # scheduler step (per-iter as in your script)
        scheduler.step(loss)

        # record
        history_loss.append(float(loss.item()))
        history_mare.append(float(mse_loss_val.item()))
        iter_times.append(time.time() - t0)

        # checkpoint logging & plotting
        if (it % checkpoint_interval == 0) or (it == num_iters - 1):
            lr_now = optimizer.param_groups[0]['lr']
            mean_iter_time = float(np.mean(iter_times[-checkpoint_interval:])) if len(iter_times) >= 1 else 0.0
            print(f"Iter {it:05d} | Total Loss: {loss.item():.6f} | MARE Loss: {mse_loss_val.item():.6f} | LR: {lr_now:.2e}")
            print(f" mean iter time (last {min(len(iter_times), checkpoint_interval)} iters): {mean_iter_time:.3f}s")

            # Save convergence plots
            plt.figure(figsize=(6,4))
            plt.plot(history_loss, label="Total Loss")
            plt.plot(history_mare, label="MARE")
            plt.yscale('log')
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Convergence")
            conv_path = os.path.join(results_dir, f"convergence_iter{it:06d}.png")
        #    plt.tight_layout()
            plt.savefig(conv_path, dpi=200)
            plt.close()

            # Results and parity plots (colored by Y error)
            P_opt = P.detach().cpu().numpy()  # (batch_size, n_params)
            with torch.no_grad():
                Q_opt_tensor = model.forward(torch.tensor(P_opt, dtype=torch.float32, device=torch_device))
            Q_opt = Q_opt_tensor.detach().cpu().numpy()

            # compute Y errors per sample
            Y_errors = np.mean(np.abs(Q_opt - Y_true) / (np.abs(Y_true) + 1e-8), axis=1)

            # Parity plots
            n_rows = int(np.ceil(n_params / 3))
            n_cols = 3
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = axes.flatten() if n_params > 1 else [axes]

            from matplotlib.colors import Normalize
            norm = Normalize(vmin=0, vmax=1)

            for ip in range(n_params):
                ax = axes[ip]
                true_params = X_true[:, ip]
                recovered_params = P_opt[:, ip]
                sc = ax.scatter(true_params, recovered_params, c=np.clip(Y_errors, 0, .1),
                                cmap="viridis", alpha=0.7, s=25, edgecolors="none")
                lims = [min(true_params.min(), recovered_params.min()),
                        max(true_params.max(), recovered_params.max())]
                ax.plot(lims, lims, "r--", linewidth=1.0)
                ax.set_xlabel(f"True Param {ip+1}")
                ax.set_ylabel(f"Recovered Param {ip+1}")
                ax.set_title(f"Param {ip+1}")

            # hide extras
            for j in range(n_params, len(axes)):
                fig.delaxes(axes[j])

            cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.95)
            cbar.set_label("Relative Y Error")
            plt.suptitle(f"Parity Plots (iter {it})", fontsize=14)
            parity_path = os.path.join(results_dir, f"parity_iter{it:06d}.png")
          #  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(parity_path, dpi=300)
            plt.close(fig)

            # Save intermediate data
            np.save(os.path.join(results_dir, f"P_opt_iter{it:06d}.npy"), P_opt)
            np.save(os.path.join(results_dir, f"Q_opt_iter{it:06d}.npy"), Q_opt)

            # Optional validation on true model (if provided)
            validation = {}
            if true_model is not None:
                try:
                    # true_model may expect (n_params, batch) transposed as earlier run_model_to_array
                    # try first true_model.predict if available
                    if hasattr(true_model, "predict"):
                        true_preds = true_model.predict(P_opt)
                    else:
                        # fallback to run_model_to_array for your full simulator interface
                        true_preds = run_model_to_array(P_opt.T, model=true_model)
                        # align shape: (batch_size, q) or (batch_size, q+1)
                        if true_preds.ndim == 2 and true_preds.shape[0] == P_opt.shape[0]:
                            pass
                        else:
                            true_preds = np.atleast_2d(true_preds)
                    # ensure same columns as Y_true if full model returns extra col
                    if true_preds.shape[1] > Y_true.shape[1]:
                        true_preds = true_preds[:, :Y_true.shape[1]]
                    mse_per_case = np.mean((true_preds - Y_true) ** 2, axis=1)
                    validation['mse_per_case'] = mse_per_case.tolist()
                except Exception as e:
                    validation['error'] = f"validation failed: {e}"

            # checkpoint metadata
            checkpoint_meta = {
                "iteration": it,
                "loss": float(loss.item()),
                "mare": float(mse_loss_val.item()),
                "lr": float(lr_now),
                "mean_iter_time": mean_iter_time,
                "validation": validation,
                "timestamp": time.time()
            }
            with open(os.path.join(results_dir, f"checkpoint_{it:06d}.json"), "w") as fh:
                json.dump(checkpoint_meta, fh, indent=2)

        # stopping criteria
        lr_now = optimizer.param_groups[0]['lr']
        if loss.item() < stop_loss:
            print(f"Stopping early at iter {it}, total loss={loss.item():.6e}")
            break
        if lr_now < lr_stop_threshold:
            print(f"Learning rate too small at iter {it} (lr={lr_now:.2e}), stopping.")
            break

    # End timing and final save
    elapsed = time.time() - t_start
    P_opt = P.detach().cpu().numpy()
    with torch.no_grad():
        Q_opt_tensor = model.forward(torch.tensor(P_opt, dtype=torch.float32, device=torch_device))
    Q_opt = Q_opt_tensor.detach().cpu().numpy()

    # Final validation on true model (if provided)
    validation_final = {}
    if true_model is not None:
        try:
            if hasattr(true_model, "predict"):
                true_preds = true_model.predict(P_opt)
            else:
                true_preds = run_model_to_array(P_opt.T, model=true_model)
                if true_preds.shape[1] > Y_true.shape[1]:
                    true_preds = true_preds[:, :Y_true.shape[1]]
            rel_err = np.abs(Q_opt - true_preds) / (np.abs(true_preds) + 1e-8)
            validation_final['mean_rel_error'] = float(np.mean(rel_err))
            validation_final['per_sample_rel_error_mean'] = np.mean(rel_err, axis=1).tolist()
        except Exception as e:
            validation_final['error'] = str(e)

    # Save final results
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "P_opt_final.npy"), P_opt)
    np.save(os.path.join(results_dir, "Q_opt_final.npy"), Q_opt)

    summary = {
        "total_runtime_sec": float(elapsed),
        "final_loss": float(history_loss[-1]) if len(history_loss) > 0 else None,
        "final_mare": float(history_mare[-1]) if len(history_mare) > 0 else None,
        "mean_iter_time": float(np.mean(iter_times)) if len(iter_times) > 0 else None,
        "batch_indices": indices.tolist(),
        "n_params": n_params,
        "batch_size": batch_size,
        "num_iters_ran": it + 1,
        "validation_final": validation_final
    }
    with open(os.path.join(results_dir, "results_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print("Optimization finished. Summary:", summary)

    return P_opt, Q_opt, summary

# ---------------------------------------------------------------------
# Example usage (keeps the same style and sectioning as your original)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # set size params (same style)
    ti = 0; tf = 1000; dt = 0.01; dtS = 1
    modelB.setSizeParameters(ti, tf, dt, dtS)

    with open("trainned_models/prob_C/nmodel_medium_5K.pth", "rb") as f:
        nmodel: NModel = pickle.load(f)
    nmodel.model.eval()

    X = pd.read_csv("data/Generated_Data_5K/ModelC/X.csv").values
    Y = pd.read_csv("data/Generated_Data_5K/ModelC/Y.csv").values[:, 0:]

    # Run optimizer (you can pass true_model=modelC for validation)
    P_opt, Q_opt, summary = inverse_problem_adam(
        model=nmodel,
        X=X,
        Y=Y,
        dist=modelC.getDist(low=0.5, high=1.5),
        batch_size=100,
        num_iters=10000,
        lr=5e-2,
        lambda_prior=1e-4,
        P_min=0.75,
        P_max=1.25,
        checkpoint_interval=1000,
        results_dir="Results/InverseProblem/NNgradinv",
        true_model=modelC  # or None if you don't want extra validation calls
    )

    # Final comparison print (if true_model provided)
    if 'validation_final' in summary and summary['validation_final']:
        print("Final validation info:", summary['validation_final'])
