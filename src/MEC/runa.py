# -----------------------------------------------------------------------------
# Unified sampling + model execution wrapper
# -----------------------------------------------------------------------------

import os
import shutil

import dolfin
from dolfin import *
import ufl

import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt

from lv_passive_filling import *
from run_passive_filling_ho_tiso_orig import LVPassiveFilling_tiso_orig


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def generate_samples(dist, ns):
    return dist.sample(ns)


def ensure_output(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------

def model_tiso(mesh, base, endo, epi, ds, nmesh, f0, s0, n0, i, params):
    a, b, af, bf = params
    return LVPassiveFilling_tiso_orig(
        mesh, base, endo, epi, ds, nmesh, f0, s0, n0, i, a, b, af, bf
    )


def model_ho8(mesh, geoparams, i, params):
    # Example for 8-parameter model wrapper
    return LVPassiveFilling(
        geoparams,
        {
            "a": params[0],
            "b": params[1],
            "af": params[2],
            "bf": params[3],
            "a_s": params[4],
            "bs": params[5],
            "afs": params[6],
            "bfs": params[7],
        },
        i,
    )


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

MODEL_REGISTRY = {
    "tiso": {
        "npar": 4,
        "sampler": lambda p: cp.J(
            cp.Uniform(0.7 * p["a0"], 1.3 * p["a0"]),
            cp.Uniform(0.7 * p["b0"], 1.3 * p["b0"]),
            cp.Uniform(0.7 * p["af0"], 1.3 * p["af0"]),
            cp.Uniform(0.7 * p["bf0"], 1.3 * p["bf0"]),
        ),
        "runner": model_tiso,
    },
    "ho8": {
        "npar": 8,
        "sampler": lambda p: cp.J(
            cp.Uniform(0.7 * p["a0"], 1.3 * p["a0"]),
            cp.Uniform(0.7 * p["b0"], 1.3 * p["b0"]),
            cp.Uniform(0.7 * p["af0"], 1.3 * p["af0"]),
            cp.Uniform(0.7 * p["bf0"], 1.3 * p["bf0"]),
            cp.Uniform(0.7 * p["as0"], 1.3 * p["as0"]),
            cp.Uniform(0.7 * p["bs0"], 1.3 * p["bs0"]),
            cp.Uniform(0.7 * p["afs0"], 1.3 * p["afs0"]),
            cp.Uniform(0.7 * p["bfs0"], 1.3 * p["bfs0"]),
        ),
        "runner": model_ho8,
    },
}


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_model(model_name, ns, output_folder="output"):

    if model_name not in MODEL_REGISTRY:
        raise ValueError("Unknown model: " + model_name)

    info = MODEL_REGISTRY[model_name]

    # Reference parameters
    params_ref = {
        "a0": 150,
        "b0": 6.0,
        "af0": 116.85,
        "bf0": 11.83425,
        "as0": 372,
        "bs0": 5.16,
        "afs0": 410,
        "bfs0": 11.3,
    }

    # Build distribution and sample
    dist = info["sampler"](params_ref)
    samples = generate_samples(dist, ns)

    # Prepare output
    ensure_output(output_folder)
    fd = os.path.join(output_folder, "data.txt")
    f_data = open(fd, "w")

    # Mesh and microstructure
    mesh, base, endo, epi, ds, nmesh = CreateMesh()
    f0, s0, n0 = CreateFiberfield(mesh)

    geoparams = {
        "mesh": mesh,
        "ds": ds,
        "nmesh": nmesh,
        "base": base,
        "endo": endo,
        "epi": epi,
        "f0": f0,
        "s0": s0,
        "n0": n0,
    }

    # Execute model evaluations
    evals = []

    for i in range(samples.shape[1]):
        Z = samples[:, i]
        runner = info["runner"]

        if model_name == "tiso":
            out = runner(mesh, base, endo, epi, ds, nmesh, f0, s0, n0, i, Z)
        else:
            out = runner(mesh, geoparams, i, Z)

        if out is not None:
            evals.append(out)
            row = np.concatenate((Z, out))
            np.savetxt(
                f_data,
                row.reshape(1, -1),
                fmt="%16.8e",
                delimiter=",",
            )
        else:
            print("Sample", i, "failed.")

    f_data.close()

    # Save all samples and evaluations
    np.savetxt(os.path.join(output_folder, "all_samples.txt"), samples)
    np.savetxt(
        os.path.join(output_folder, "all_evals.txt"),
        np.array(evals).T,
    )


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

run_model("tiso", 120, "outputa")
run_model("ho8", 200, "outputb")
