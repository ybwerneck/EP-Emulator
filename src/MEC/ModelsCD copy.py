# -----------------------------------------------------------------------------
# Unified sampling + model execution wrapper
# -----------------------------------------------------------------------------
from dolfin import *      # All PDE classes, mesh, functions, solvers

import numpy as np

# Optional, frequently needed
import mpi4py.MPI as PMPI
import matplotlib.pyplot as plt
from lv_passive_filling import *
from run_passive_filling_ho_tiso_orig import LVPassiveFilling_tiso_orig

import logging
import os
import sys
from mpi4py import MPI

def setup_mpi_logger(name="solver", logdir="logs"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    os.makedirs(logdir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False   # critical under MPI

    formatter = logging.Formatter(
        fmt="%(asctime)s | rank %(rank)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    # File handler (one per rank)
    fh = logging.FileHandler(os.path.join(logdir, f"{name}_rank{rank:03d}.log"))
    fh.setFormatter(formatter)
    fh.addFilter(RankFilter())
    logger.addHandler(fh)

    # Console handler (rank 0 only)
    if rank == 0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.addFilter(RankFilter())
        logger.addHandler(sh)

    return logger

from dolfin import *
import numpy as np
import os
import shutil
import mpi4py.MPI as PMPI
import matplotlib.pyplot as plt

from lv_passive_filling import *
from run_passive_filling_ho_tiso_orig import LVPassiveFilling_tiso_orig

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
from mpi4py import MPI
logger = setup_mpi_logger("lv_sampling", logdir="logs")

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

# ------------------------------------------------------------------
# Model wrappers
# ------------------------------------------------------------------
def model_tiso(geoparams,params,i):
    a, b, af, bf = params
    return LVPassiveFilling_tiso_orig(
        geoparams,
        {'a':params[0],'b':params[1],'af':params[2],'bf':params[3]},
        i
    )

def model_ho8(geoparams,params,i):
    return LVPassiveFilling(
        geoparams,
        {'a':params[0],'b':params[1],'af':params[2],'bf':params[3],
         'a_s':params[4],'bs':params[5],'afs':params[6],'bfs':params[7]},
        i
    )

MODEL_REGISTRY = {
    "tiso": {
        "npar": 4,
        "sampler": lambda p: cp.J(
            cp.Uniform(0.7*p['a0'],1.3*p['a0']),
            cp.Uniform(0.7*p['b0'],1.3*p['b0']),
            cp.Uniform(0.7*p['af0'],1.3*p['af0']),
            cp.Uniform(0.7*p['bf0'],1.3*p['bf0'])
        ),
        "runner": model_tiso
    },
    "ho8": {
        "npar": 8,
        "sampler": lambda p: cp.J(
            cp.Uniform(0.7*p['a0'],1.3*p['a0']),
            cp.Uniform(0.7*p['b0'],1.3*p['b0']),
            cp.Uniform(0.7*p['af0'],1.3*p['af0']),
            cp.Uniform(0.7*p['bf0'],1.3*p['bf0']),
            cp.Uniform(0.7*p['as0'],1.3*p['as0']),
            cp.Uniform(0.7*p['bs0'],1.3*p['bs0']),
            cp.Uniform(0.7*p['afs0'],1.3*p['afs0']),
            cp.Uniform(0.7*p['bfs0'],1.3*p['bfs0'])
        ),
        "runner": model_ho8
    }
}

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def run_model(model_name, ns, output_folder="output"):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger.info(f"Starting model '{model_name}' with ns={ns}")

    if model_name not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_name}")
        raise ValueError(model_name)

    info = MODEL_REGISTRY[model_name]

    params_ref = {
        'a0':150, 'b0':6.0, 'af0':116.85, 'bf0':11.83425,
        'as0':372, 'bs0':5.16, 'afs0':410, 'bfs0':11.3
    }

    dist = info["sampler"](params_ref)
    samples = generate_samples(dist, ns)

    if rank == 0:
        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
            except:
                print("Could not remove existing output folder")
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder ready: {output_folder}")

    comm.Barrier()

    fd = os.path.join(output_folder, f"data_rank{rank:03d}.txt")
    f_data = open(fd, "w")

    logger.info("Building mesh and fiber field")
    mesh, base, endo, epi, ds, nmesh = CreateMesh()
    f0, s0, n0 = CreateFiberfield(mesh)

    geoparams = {'mesh': mesh, 
                 'base': base, 'endo': endo, 'epi': epi, 
	             'ds': ds, 'nmesh': nmesh, 
                 'f0': f0, 's0': s0, 'n0': n0}

    runner = info["runner"]

    for i in range(rank, samples.shape[1], size):
        Z = samples[:, i]
        logger.info(f"Running sample {i}")

        try:
            if model_name == "tiso":
                out = runner(geoparams, Z, i)
            else:
                out = runner(geoparams, Z, i)

            if out is not None:
                row = np.concatenate((Z, out))
               # np.savetxt(f_data, row.reshape(1,-1),
                        #   fmt="%16.8e", delimiter=",")
            else:
                logger.warning(f"Sample {i} returned None")

        except Exception:
            logger.exception(f"Sample {i} failed")

    f_data.close()
    logger.info("Local sampling finished")

    comm.Barrier()

    if rank == 0:
        logger.info("Merging outputs")
        all_rows = []
        for r in range(size):
            fname = os.path.join(output_folder, f"data_rank{r:03d}.txt")
            if os.path.exists(fname):
                data = np.loadtxt(fname, delimiter=",")
                if data.ndim == 1:
                    data = data.reshape(1,-1)
                all_rows.append(data)

        if all_rows:
            all_data = np.vstack(all_rows)
            np.savetxt(os.path.join(output_folder, "data_all.txt"),
                       all_data, fmt="%16.8e", delimiter=",")
            logger.info("Merged output written")

