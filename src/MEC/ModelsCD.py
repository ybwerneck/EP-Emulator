# -----------------------------------------------------------------------------
# Unified sampling + model execution wrapper
# -----------------------------------------------------------------------------
from dolfin import *      # All PDE classes, mesh, functions, solvers

import numpy as np
import sys
import mpi4py.MPI as PMPI
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.MEC.lv_passive_filling import * ##FULL
from src.MEC.run_passive_filling_ho_tiso_orig import LVPassiveFilling_tiso_orig ##TISO



MPI= PMPI
# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def generate_samples(dist, ns):
    return dist.sample(ns)

def ensure_output(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

MPI = PMPI




# =============================================================================
# Model wrappers (unchanged numerics)
# =============================================================================

def model_tiso(geoparams, params, sample_id):
    return LVPassiveFilling_tiso_orig(
        geoparams,
        {
            'a':  params[0],
            'b':  params[1],
            'af': params[2],
            'bf': params[3],
        },
        sample_id
    )


def model_ho8(geoparams, params, sample_id):
    return LVPassiveFilling(
        geoparams,
        {
            'a':   params[0], 'b':   params[1],
            'af':  params[2], 'bf':  params[3],
            'a_s': params[4], 'bs':  params[5],
            'afs': params[6], 'bfs': params[7],
        },
        sample_id
    )



class TisoModel():
    npar = 4
    nout = 6
    
    params_ref = {
        'a0': 150, 'b0': 6.0, 'af0': 116.85, 'bf0': 11.83425,
    }

    def get_dist(self, low=0.7, high=1.3):
        """Return joint distribution for multiplicative coefficients."""
        return cp.J(
            cp.Uniform(low, high),
            cp.Uniform(low, high),
            cp.Uniform(low, high),
            cp.Uniform(low, high),
        )

    info = {"npar": 4, "nout": 6}

    def run_model(self, X):
        """
        Run model where X contains multiplicative coefficients for parameters.
        X shape: (npar, nsamples)
        """
        coefs = np.array([self.params_ref[k] for k in ['a0','b0','af0','bf0']]).reshape(-1,1)
        X_scaled = X * coefs  # multiply each row by its parameter
        return run_model(model_tiso, "tiso", self.info, X_scaled)


class Ho8Model():
    npar = 8
    nout = 6

    params_ref = {
        'a0': 150,  'b0': 6.0,  'af0': 116.85, 'bf0': 11.83425,
        'as0': 372, 'bs0': 5.16, 'afs0': 410,  'bfs0': 11.3,
    }

    def get_dist(self, low=0.7, high=1.3):
        """Return joint distribution for multiplicative coefficients."""
        p = self.params_ref
        a0=cp.Uniform(low, high)
        b0=cp.Uniform(low, high)
        af0=cp.Uniform(low, high)
        bf0=cp.Uniform(low, high)
        as0=cp.Uniform(low, high)
        bs0=cp.Uniform(low, high)
        afs0=cp.Uniform(low, high)
        bfs0=cp.Uniform(low, high)
        return cp.J(a0, b0, af0, bf0, as0, bs0, afs0, bfs0)

    info = {"npar": 8, "nout": 6}

    def run_model(self, X):
        """
        Run model where X contains multiplicative coefficients for parameters.
        X shape: (npar, nsamples)
        """
        coefs = np.array([self.params_ref[k] for k in ['a0','b0','af0','bf0','as0','bs0','afs0','bfs0']]).reshape(-1,1)
        X_scaled = X * coefs
        return run_model(model_ho8, "ho8", self.info, X_scaled)

# =============================================================================
# Main driver (X → Y) — process-level parallelism, serial FEniCS
# =============================================================================
import numpy as np
import logging
from multiprocessing import Pool


    # ------------------------------------------------------------------
    # Geometry (built once per process)
    # ------------------------------------------------------------------
mesh, base, endo, epi, ds, nmesh = CreateMesh()
f0, s0, n0 = CreateFiberfield(mesh)

def _run_single_sample(args):
    """
    Worker function: runs exactly one sample in a fresh process.
    """
    runner, model_name, info, x, i = args
    print("\n\n\n\----------------------------------------------------------------\n\n\n",flush=True)
    print(f"Running sample {i} for model '{model_name}'",flush=True)
    print("\n\n\n\----------------------------------------------------------------\n\n\n",flush=True)


    geoparams = {
        'mesh': mesh,
        'base': base, 'endo': endo, 'epi': epi,
        'ds': ds, 'nmesh': nmesh,
        'f0': f0, 's0': s0, 'n0': n0,
    }

    try:
        out = runner(geoparams, x, i)
        if out is None:
            print(f"Sample {i} returned None")
        return i, out
    except Exception:
        print(f"Sample {i} failed")
        return i, None


def run_model(runner, model_name, info, X, nproc=1):
    """
    Runs a model using external process-level parallelism and returns outputs.

    Parameters
    ----------
    runner : callable
        Model function, signature (geoparams, x, i) -> output vector
    model_name : str
        Label for logging
    info : dict
        Model info, must contain 'npar'
    X : ndarray, shape (npar, nsamples)
        Input parameter samples
    nproc : int or None
        Number of worker processes (defaults to os.cpu_count())

    Returns
    -------
    Y : ndarray, shape (nout, nsamples)
        Computed outputs
    """

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if X.ndim != 2:
        raise ValueError("X must have shape (npar, nsamples)")

    if X.shape[0] != info["npar"]:
        raise ValueError(
            f"Model '{model_name}' expects {info['npar']} parameters, "
            f"got {X.shape[0]}"
        )

    ns = X.shape[1]
    print(f"Starting model '{model_name}' with {ns} samples")

    nout = info.get("nout", 6)
    Y = np.zeros((nout, ns), dtype=float)

    # ------------------------------------------------------------------
    # Prepare tasks
    # ------------------------------------------------------------------
    tasks = [
        (runner, model_name, info, X[:, i], i)
        for i in range(ns)
    ]

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    with Pool(processes=nproc) as pool:
        results = pool.map(_run_single_sample, tasks)

    # ------------------------------------------------------------------
    # Gather results
    # ------------------------------------------------------------------
    for i, out in results:
        if out is not None:
            Y[:, i] = out

    print("All samples computed and gathered")
    return Y


if __name__ == "__main__":

    # Simple test: generate 5 random samples per model
    n_samples = 2

    # TISO
    tiso = TisoModel()
    X_tiso = tiso.get_dist(low=0.7, high=1.3).sample(n_samples)
    Y_tiso = tiso.run_model(X_tiso)
    if Y_tiso is not None:
        print("TISO outputs:", Y_tiso)

    # HO8
    ho8 = Ho8Model()
    X_ho8 = ho8.get_dist(low=0.7, high=1.3).sample(n_samples)
    Y_ho8 = ho8.run_model(X_ho8)
    if Y_ho8 is not None:
        print("HO8 outputs:", Y_ho8)
