# ------------- begin ttnnp2004_njit_generated.py -------------
import os
os.environ["NUMBA_CUDA_PTX_VERSION"] = "84"

import numpy as np
from numba import njit, prange

# Auto-generated Ten Tusscher 2004 (TNNP2004b) full-model NJIT solver
# Generated from uploaded CellML. This file implements rhs and an Euler integrator.
from numba import cuda, float32
import math
import numpy as np
from numba import njit

# --- Constants / indices ---
N_STATES = 18
N_STATES_HH = 12
N_ALGS = 68
NL_START = 0
NL_END = 6
HH_START = 6
EPSILON = 1e-12

# --- State indices (matching #define mapping) ---
V_idx      = 0
Ca_i_idx   = 1
Ca_SR_idx  = 2
Na_i_idx   = 3
K_i_idx    = 4
ikatp_idx  = 5
Xr1_idx = 6
Xr2_idx = 7
Xs_idx  = 8
m_idx   = 9
h_idx   = 10
j_idx   = 11
d_idx   = 12
f_idx   = 13
fCa_idx = 14
s_idx   = 15
r_idx   = 16
g_idx   = 17

# Stimulus parameters
stim_state = 0
stim_amplitude = -52.0
stim_period = 1.0e3
stim_start = 5.0
stim_duration = 1.0

# Physical constants
R       = 8.3144720e3
T       = 3.10e2
F       = 9.64853415e4
Na_o    = 1.40e2
P_kna   = 3.0e-2
Ca_o    = 2.0e0
      # 0.270

g_bna   = 2.90e-4

P_NaK   = 1.3620e0
K_mk    = 1.0e0
K_NaCa  = 1.0e3
gamma   = 3.50e-1
alpha   = 2.50e0
Km_Nai  = 8.750e1
Km_Ca   = 1.380e0
K_sat   = 1.0e-1
g_pCa   = 8.250e-1
K_pCa   = 5.0e-4
g_pK    = 1.460e-2
a_rel   = 1.64640e-2
b_rel   = 2.50e-1
c_rel   = 8.2320e-3
Vmax_up = 4.250e-4
K_up    = 2.50e-4
V_leak  = 8.0e-5
tau_g   = 2.0e0
Buf_c   = 1.50e-1
K_buf_c = 1.0e-3
Buf_sr  = 1.0e1
K_buf_sr= 3.0e-1
V_c     = 1.64040e-2
Cm      = 1.850e-1
V_sr    = 1.0940e-3
K_mNa   = 4.0e1

##Parameters indexes

G_NA_IDX   = 0
G_CAL_IDX  = 1
K_I_IDX    = 2
K_O_IDX    = 3
ATP_IDX    = 4
G_K1_IDX   = 5
G_KR_IDX   = 6
G_KS_IDX   = 7
G_TO_IDX   = 8
G_BCA_IDX  = 9
G_PK_IDX   = 10
G_PCA_IDX  = 11


### Algebraic variable indices
I_STIM       = 0
E_NA         = 1
E_K          = 2
E_KS         = 3
E_CA         = 4
ALPHA_K1     = 5
BETA_K1      = 6
XK1_INF      = 7
I_K1         = 8
I_KR         = 9
XR1_INF      = 10
ALPHA_XR1    = 11
BETA_XR1     = 12
TAU_XR1      = 13
XR2_INF      = 14
ALPHA_XR2    = 15
BETA_XR2     = 16
TAU_XR2      = 17
I_KS         = 18
XS_INF       = 19
ALPHA_XS     = 20
BETA_XS      = 21
TAU_XS       = 22
I_NA         = 23
M_INF        = 24
ALPHA_M      = 25
BETA_M       = 26
TAU_M        = 27
H_INF        = 28
ALPHA_H      = 29
BETA_H       = 30
TAU_H        = 31
J_INF        = 32
ALPHA_J      = 33
BETA_J       = 34
TAU_J        = 35
I_B_NA       = 36
I_CaL        = 37
D_INF        = 38
ALPHA_D      = 39
BETA_D       = 40
GAMMA_D      = 41
TAU_D        = 42
F_INF        = 43
TAU_F        = 44
ALPHA_FCA    = 45
BETA_FCA     = 46
GAMA_FCA     = 47
FCA_INF      = 48
TAU_FCA      = 49
D_FCA        = 50
I_B_CA       = 51
I_TO         = 52
S_INF        = 53
TAU_S        = 54
R_INF        = 55
TAU_R        = 56
I_NaK        = 57
I_NaCa       = 58
I_P_Ca       = 59
I_P_K        = 60
I_REL        = 61
I_UP         = 62
I_LEAK       = 63
G_INF        = 64
D_G          = 65
CA_I_BUFC    = 66
CA_SR_BUFSR  = 67


V_IDX = 0
CA_I_IDX = 1
FCA_IDX = 14
G_IDX = 17

TAU_G = 2.0  # as in original



from numba import cuda, float32

from numba import cuda

from numba import cuda, float32
import numpy as np
def run_model_njit_gpu(param_matrix, ti=0.0, tf=1000.0, dt=0.01, dtS=0.1):
    """
    Run TNNP2004 model on GPU and track variable evolution over time.

    Parameters
    ----------
    param_matrix : float32[n_threads, n_params]
        Each row is a parameter set for one simulation.
    ti : float
        Start recording time (ms)
    tf : float
        End time (ms)
    dt : float
        Integration timestep (ms)
    dtS : float or None
        Sampling interval (ms). If None, only final state is returned.

    Returns
    -------
    Y_time : float32[n_threads, n_samples, N_STATES]
        State evolution over time (from ti to tf)
    """
    n_threads = param_matrix.shape[0]
    args_flat = param_matrix.T.flatten().astype(np.float32)

    # Total integration steps
    n_steps = int(tf / dt)

    # Number of samples (only after ti)
    if dtS is None:
        n_samples = 1
    else:
        n_samples = int((tf - ti) / dtS) + 1

    # Allocate output array
    Y_time = np.zeros((n_threads, n_samples, N_STATES), dtype=np.float32)

    # CUDA grid
    threads_per_block = 256
    blocks_per_grid = (n_threads + threads_per_block - 1) // threads_per_block

    # Launch kernel
    gpu_simulate[blocks_per_grid, threads_per_block](
        Y_time, param_matrix, dt, n_steps, dtS if dtS is not None else dt * n_steps, ti, tf
    )

    cuda.synchronize()
    return Y_time


@cuda.jit
def gpu_simulate(Y_time, args, dt, n_steps, dtS, ti, tf):
    tid = cuda.grid(1)
    if tid >= Y_time.shape[0]:
        return

    Y_old = cuda.local.array(N_STATES, dtype=float32)
    Y_new = cuda.local.array(N_STATES, dtype=float32)
    pars  = cuda.local.array(12, dtype=float32)
    algs  = cuda.local.array(N_ALGS, dtype=float32)
    rhs   = cuda.local.array(N_STATES, dtype=float32)

    # Initialize states
    gpu_init_model(Y_old, pars, args, Y_time.shape[0])

    t = 0.0
    sample_idx = 0
    next_sample = ti  # <-- start recording only after ti

    for step in range(n_steps):
        # One integration step
        unified_step(Y_new, pars, algs, rhs, Y_old, t, dt)

        for i in range(N_STATES):
            Y_old[i] = Y_new[i]

        t += dt

        # Save state only if past ti and aligned with dtS
        if t >= next_sample - 1e-8 and t <= tf - 1e-8:
            for i in range(N_STATES):
                Y_time[tid, sample_idx, i] = Y_old[i]
            sample_idx += 1
            next_sample += dtS


from numba import cuda, float32
import numpy as np
@cuda.jit(device=True)
def log(x):
    return math.log(x)
@cuda.jit(device=True)
def sqrt(x):
    return math.sqrt(x)
@cuda.jit(device=True)
def floor(x):
    return math.floor(x)
@cuda.jit(device=True)
def pow(x,y):
    return math.pow(x,y)
@cuda.jit
def gpu_init_model(Y_old,pars, args, N):
    tid = cuda.grid(1)
    if tid >=N:
        return


    # --- Load per-thread parameters into local pars ---]
    pars[G_NA_IDX]   = args[tid , G_NA_IDX ]   # g_Na
    pars[G_CAL_IDX]  = args[tid , G_CAL_IDX]   # g_CaL
    pars[K_I_IDX]    = args[tid , K_I_IDX ]   # K_i
    pars[K_O_IDX]    = args[tid , K_O_IDX ]   # K_o
    pars[ATP_IDX]    = args[tid , ATP_IDX ]   # atp
    pars[G_K1_IDX]   = args[tid , G_K1_IDX ]   # g_K1
    pars[G_KR_IDX]   = args[tid , G_KR_IDX ]   # g_Kr
    pars[G_KS_IDX]   = args[tid , G_KS_IDX]   # g_Ks
    pars[G_TO_IDX]   = args[tid ,G_TO_IDX]   # g_to
    pars[G_BCA_IDX]  = args[tid ,G_BCA_IDX ]   # g_bca
    pars[G_PK_IDX]   = args[tid , G_PK_IDX ]  # g_pK
    pars[G_PCA_IDX]  = args[tid , G_PCA_IDX ] / (1.0 + (1.4 / pars[ATP_IDX])**2.6)  # g_pCa corrected

    # --- Initialize states ---
    Y_old[V_idx]      = -86.2
    Y_old[Ca_i_idx]   = 8e-5
    Y_old[Ca_SR_idx]  = 0.56
    Y_old[Na_i_idx]   = 11.6
    Y_old[K_i_idx]    = pars[K_I_IDX]  
    Y_old[ikatp_idx]  = 0.0

    Y_old[Xr1_idx] = 0.0
    Y_old[Xr2_idx] = 1.0
    Y_old[Xs_idx]  = 0.0
    Y_old[m_idx]   = 0.0
    Y_old[h_idx]   = 0.75
    Y_old[j_idx]   = 0.75
    Y_old[d_idx]   = 0.0
    Y_old[f_idx]   = 1.0
    Y_old[fCa_idx] = 1.0
    Y_old[s_idx]   = 1.0
    Y_old[r_idx]   = 0.0
    Y_old[g_idx]   = 1.0



@cuda.jit(device=True)
def exp(x):
    return math.exp(x)
@cuda.jit(device=True)
def calc_algs_hh(algs, Y_old, t):
    V = Y_old[V_IDX]
    Ca_i = Y_old[CA_I_IDX]
    fCa_old = Y_old[FCA_IDX]
    g_old = Y_old[G_IDX]

    # --- XR1 ---
    algs[XR1_INF] = 1.0 / (1.0 + exp((-26.0 - V)/7.0))
    algs[ALPHA_XR1] = 450.0 / (1.0 + exp((-45.0 - V)/10.0))
    algs[BETA_XR1] = 6.0 / (1.0 + exp((V + 30)/11.5))
    algs[TAU_XR1] = 1.0 * algs[ALPHA_XR1] * algs[BETA_XR1]

    # --- XR2 ---
    algs[XR2_INF] = 1.0 / (1.0 + exp((V + 88)/24.0))
    algs[ALPHA_XR2] = 3.0 / (1.0 + exp((-60 - V)/20.0))
    algs[BETA_XR2] = 1.12 / (1.0 + exp((V - 60)/20.0))
    algs[TAU_XR2] = 1.0 * algs[ALPHA_XR2] * algs[BETA_XR2]

    # --- XS ---
    algs[XS_INF] = 1.0 / (1.0 + exp((-5.0 - V)/14.0))
    algs[ALPHA_XS] = 1100.0 / pow(1.0 + exp((-10 - V)/6.0), 0.5)
    algs[BETA_XS] = 1.0 / (1.0 + exp((V - 60)/20.0))
    algs[TAU_XS] = 1.0 * algs[ALPHA_XS] * algs[BETA_XS]

    # --- M gate ---
    algs[M_INF] = 1.0 / pow(1.0 + exp((-56.86 - V)/9.03), 2)
    algs[ALPHA_M] = 1.0 / (1.0 + exp((-60 - V)/5.0))
    algs[BETA_M] = 0.1/(1.0+exp((V+35)/5)) + 0.1/(1.0+exp((V-50)/200))
    algs[TAU_M] = 1.0 * algs[ALPHA_M] * algs[BETA_M]

    # --- H gate ---
    algs[H_INF] = 1.0 / pow(1.0 + exp((V + 71.55)/7.43), 2)
    if V < -40:
        algs[ALPHA_H] = 0.057 * exp(-(V + 80)/6.8)
        algs[BETA_H] = 2.7 * exp(0.079*V) + 310000.0 * exp(0.3485*V)
    else:
        algs[ALPHA_H] = 0.0
        algs[BETA_H] = 0.77 / (0.13*(1+exp((V+10.66)/-11.1)))
    algs[TAU_H] = 1.0 / (algs[ALPHA_H] + algs[BETA_H])

    # --- J gate ---
    algs[J_INF] = 1.0 / pow(1.0 + exp((V+71.55)/7.43), 2)
    if V < -40:
        algs[ALPHA_J] = ((-25428.0)*exp(0.2444*V) - 6.948e-6*exp(-0.04391*V))*(V+37.78)/(1+exp(0.311*(V+79.23)))
        algs[BETA_J] = 0.02424*exp(-0.01052*V)/(1+exp(-0.1378*(V+40.14)))
    else:
        algs[ALPHA_J] = 0.0
        algs[BETA_J] = 0.6*exp(0.057*V)/(1+exp(-0.1*(V+32)))
    algs[TAU_J] = 1.0 / (algs[ALPHA_J] + algs[BETA_J])

    # --- D gate ---
    algs[D_INF] = 1.0/(1.0 + exp((-5 - V)/7.5))
    algs[ALPHA_D] = 1.4/(1.0 + exp((-35 - V)/13)) + 0.25
    algs[BETA_D] = 1.4 / (1.0 + exp((V+5)/5))
    algs[GAMMA_D] = 1.0/(1.0 + exp((50 - V)/20))
    algs[TAU_D] = 1.0*algs[ALPHA_D]*algs[BETA_D] + algs[GAMMA_D]

    # --- F gate ---
    algs[F_INF] = 1.0 / (1.0 + exp((V + 20)/7))
    algs[TAU_F] = 1125*exp(-(V+27)*(V+27)/240) + 80 + 165/(1.0 + exp((25 - V)/10))

    # --- F_Ca ---
    algs[TAU_FCA] = 2.0
    algs[ALPHA_FCA] = 1.0 / (1.0 + pow(Ca_i/0.000325,8))
    algs[BETA_FCA] = 0.1 / (1.0 + exp((Ca_i-0.0005)/0.0001))
    algs[GAMA_FCA] = 0.2 / (1.0 + exp((Ca_i-0.00075)/0.0008))
    algs[FCA_INF] = (algs[ALPHA_FCA]+algs[BETA_FCA]+algs[GAMA_FCA]+0.23)/1.46
    algs[D_FCA] = (algs[FCA_INF]-fCa_old)/algs[TAU_FCA]

    # --- S gate ---
    algs[S_INF] = 1.0 / (1.0 + exp((V+20)/5))
    algs[TAU_S] = 85*exp(-((V+45)**2)/320) + 5/(1+exp((V-20)/5)) + 3

    # --- R gate ---
    algs[R_INF] = 1.0 / (1.0 + exp((20-V)/6))
    algs[TAU_R] = 9.5*exp(-((V+40)**2)/1800) + 0.8

    # --- G ---
    if Ca_i < 3.5e-4:
        algs[G_INF] = 1.0 / (1 + pow(Ca_i/3.5e-4, 6))
    else:
        algs[G_INF] = 1.0 / (1 + pow(Ca_i/3.5e-4, 16))
    algs[D_G] = (algs[G_INF]-g_old)/TAU_G




@cuda.jit(device=True)
def calc_hh_coeff(a, b,  algs, Y_old_, t):
    # First, compute all the algebraic HH quantities
    calc_algs_hh(algs,  Y_old_, t)

    # --- a coefficients (time constants) ---
    a[0] = -1.0 / algs[13]   # Xr1_a_
    a[1] = -1.0 / algs[17]   # Xr2_a_
    a[2] = -1.0 / algs[22]   # Xs_a_
    a[3] = -1.0 / algs[27]   # m_a_
    a[4] = -1.0 / algs[31]   # h_a_
    a[5] = -1.0 / algs[35]   # j_a_
    a[6] = -1.0 / algs[42]   # d_a_
    a[7] = -1.0 / algs[44]   # f_a_
    a[8] = -1.0 / algs[49]   # fCa_a_
    a[9] = -1.0 / algs[54]   # s_a_
    a[10] = -1.0 / algs[56]  # r_a_
    a[11] = -1.0 / 2.0       # g_a_

    # --- b coefficients (steady-state terms) ---
    b[0] = algs[10] / algs[13]   # Xr1_b_
    b[1] = algs[14] / algs[17]   # Xr2_b_
    b[2] = algs[19] / algs[22]   # Xs_b_
    b[3] = algs[24] / algs[27]   # m_b_
    b[4] = algs[28] / algs[31]   # h_b_
    b[5] = algs[32] / algs[35]   # j_b_
    b[6] = algs[38] / algs[42]   # d_b_
    b[7] = algs[43] / algs[44]   # f_b_
    b[8] = algs[48] / algs[49]   # fCa_b_
    b[9] = algs[53] / algs[54]   # s_b_
    b[10] = algs[55] / algs[56]  # r_b_
    b[11] = algs[64] / 2.0     # g_b_

    # --- special conditions ---
    if (Y_old_[14] * a[8] + b[8] > 0.0) and (Y_old_[0] > -37.0):
        a[8] = 0.0
        b[8] = 0.0

    if (Y_old_[17] * a[11] + b[11] > 0.0) and (Y_old_[0] > -37.0):
        a[11] = 0.0
        b[11] = 0.0

@cuda.jit(device=True)
def calc_stimulus(pars, t):
    # --- Stimulus parameters ---
    stim_state = 0
    stim_amplitude = -52.0
    stim_period = 1.0e3
    stim_start = 5.0
    stim_duration = 1.5
    if stim_state < 0:
        return 0.0
    if stim_state > 0:
        return stim_amplitude

    t_since_last_tick = t - floor(t / stim_period) * stim_period
    pulse_end = stim_start + stim_duration
    if stim_start <= t_since_last_tick <= pulse_end:
        return stim_amplitude
    return 0.0

@cuda.jit(device=True)
def calc_algs_nl(algs, Y_old, pars, t):
    # --- States ---
    V = Y_old[V_idx]
    Ca_i = Y_old[Ca_i_idx]
    Ca_SR = Y_old[Ca_SR_idx]
    Na_i = Y_old[Na_i_idx]
    K_i = Y_old[K_i_idx]

    Xr1 = Y_old[Xr1_idx]
    Xr2 = Y_old[Xr2_idx]
    Xs  = Y_old[Xs_idx]
    m   = Y_old[m_idx]
    h   = Y_old[h_idx]
    j   = Y_old[j_idx]
    d   = Y_old[d_idx]
    f   = Y_old[f_idx]
    fCa = Y_old[fCa_idx]
    s   = Y_old[s_idx]
    r   = Y_old[r_idx]
    g   = Y_old[g_idx]

    # --- Stimulus ---
    algs[I_STIM] = calc_stimulus(pars, t)

    # --- Reversal potentials ---
    algs[E_NA] = (R * T / F) * log(Na_o / Na_i)
    algs[E_K]  = (R * T / F) * log(pars[K_O_IDX] / K_i)
    algs[E_KS] = (R * T / F) * log((pars[K_O_IDX] + P_kna * Na_o) / (K_i + P_kna * Na_i))
    algs[E_CA] = (0.5 * R * T / F) * log(Ca_o / Ca_i)

    # --- K1 gating ---
    alpha_K1 = 0.1 / (1.0 + exp(0.06 * ((V - algs[E_K]) - 200)))
    beta_K1  = (3.0 * exp(0.0002 * ((V - algs[E_K]) + 100)) + 1.0 * exp(0.1 * ((V - algs[E_K]) - 10))) / \
               (1.0 + exp(-0.5 * (V - algs[E_K])))
    algs[ALPHA_K1] = alpha_K1
    algs[BETA_K1]  = beta_K1
    algs[XK1_INF]  = alpha_K1 / (alpha_K1 + beta_K1)

    # --- Currents ---
    algs[I_K1]  = pars[G_K1_IDX] * algs[XK1_INF] * sqrt(pars[K_O_IDX] / 5.4) * (V - algs[E_K])
    algs[I_KR]  = pars[G_KR_IDX] * sqrt(pars[K_O_IDX] / 5.4) * Xr1 * Xr2 * (V - algs[E_K])
    algs[XR1_INF] = 1.0 / (1.0 + exp((-26 - V) / 7.0))

    # L-type Ca current
    algs[I_CaL] = ((pars[G_CAL_IDX] * d * f * fCa * 4.0 * V * F**2) / (R*T)) * \
                  ((Ca_i * exp(2.0 * V * F / (R*T))) - 0.341 * Ca_o) / \
                  (exp(2.0 * V * F / (R*T)) - 1.0)

    # Na/K pump
    algs[I_NaK] = ((P_NaK * pars[K_O_IDX] / (pars[K_O_IDX] + K_mk) * Na_i) / (Na_i + K_mNa)) / \
                  (1.0 + 0.1245 * exp(-0.1 * V * F / (R*T)) + 0.0353 * exp(-V * F / (R*T)))

    # Na/Ca exchanger
    algs[I_NaCa] = (K_NaCa * (exp(gamma * V * F / (R*T)) * Na_i**3 * Ca_o - \
                               exp((gamma-1.0) * V * F / (R*T)) * Na_o**3 * Ca_i * alpha)) / \
                    ((Km_Nai**3 + Na_o**3) * (Km_Ca + Ca_o) * (1.0 + K_sat * exp((gamma-1.0) * V * F / (R*T))))

    # Ca pump and SR fluxes
    algs[I_P_Ca] = pars[G_PCA_IDX] * Ca_i / (Ca_i + K_pCa)
    algs[I_TO]   = pars[G_TO_IDX] * r * s * (V - algs[E_K])
    algs[I_P_K]  = (pars[G_PK_IDX] * (V - algs[E_K])) / (1.0 + exp((25 - V)/5.98))

  
    # SR release
    algs[I_REL] = ((a_rel * Ca_SR**2 / (b_rel**2 + Ca_SR**2) + c_rel) * d * g)

    # SR uptake
    algs[I_UP] = Vmax_up / (1.0 + (K_up**2 / Ca_i**2))

    # SR leak
    algs[I_LEAK] = V_leak * (Ca_SR - Ca_i)
    # Ca buffering
    algs[CA_I_BUFC]   = 1.0 / (1.0 + (Buf_c * K_buf_c) / (Ca_i + K_buf_c)**2)
    algs[CA_SR_BUFSR] = 1.0 / (1.0 + (Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr)**2)


@cuda.jit(device=True)
def calc_rhs_nl(rhs, Y_old, algs, pars, t):
    # --- ATP-dependent K current ---
    patp = 1e6 / (1.0 + pow(pars[ATP_IDX] / 0.25, 2.0))
    gkatp_f = patp * (195e-6 / 5e3) * pow(pars[K_O_IDX] / 5.4, 0.24)
    ikatp_f = gkatp_f * (Y_old[V_idx] - 5e3)
    calc_algs_nl(algs, Y_old, pars, t)
    # --- Voltage derivative ---
    rhs[V_idx] = -(
        -ikatp_f
        + algs[I_K1]
        + algs[I_KR]
        + algs[I_KS]
        + algs[I_CaL]
        + algs[I_NaK]
        + algs[I_NA]
        + algs[I_B_NA]
        + algs[I_NaCa]
         + algs[I_P_K]
        + algs[I_P_Ca]
        + algs[I_STIM]
    )
   
    denom = 2.0 * V_c * F
    if abs(denom) < 1e-8:
        denom = 1e-8

  
    rhs[Ca_i_idx] = algs[CA_I_BUFC] * (
    (algs[I_LEAK] - algs[I_UP] + algs[I_REL])  # fluxes
    - ((algs[I_CaL] + algs[I_B_CA] + algs[I_P_Ca] - 2.0 * algs[I_NaCa]) / (2.0 * V_c * F) * Cm)  # Ca²⁺ currents
)
    rhs[Ca_SR_idx] = ((algs[CA_SR_BUFSR] * V_c) / V_sr) * (algs[I_UP] - (algs[I_REL] + algs[G_INF]))

    # --- Na_i derivative ---
    rhs[Na_i_idx] = -((algs[I_NA] + algs[I_B_NA] + 3.0 * algs[I_NaK] + 3.0 * algs[I_NaCa]) * Cm) / (V_c * F)

    # --- K_i derivative ---
    rhs[K_i_idx] = -((algs[I_K1] + algs[I_TO] + algs[I_KR] + algs[I_KS] + algs[I_P_K] + algs[I_STIM] - 2.0 * algs[I_NaK]) * Cm) / (V_c * F)


# --- Unified step: RL + Euler ---
@cuda.jit(device=True)
def unified_step(Y_new, pars, algs, rhs, Y_old, t, dt):
    
    # --- 1. HH gating step ---
    a_HH = cuda.local.array(N_STATES_HH, dtype=float32)

    b_HH = cuda.local.array(N_STATES_HH, dtype=float32)
    calc_hh_coeff(a_HH, b_HH,  algs, Y_old, t)
    
    for i in range(N_STATES_HH):
        idx = HH_START + i
        if abs(a_HH[i]) < EPSILON:
            Y_new[idx] = Y_old[idx] + dt * (a_HH[i] * Y_old[idx] + b_HH[i])
        else:
            aux = b_HH[i] / a_HH[i]
            Y_new[idx] = exp(a_HH[i] * dt) * (Y_old[idx] + aux) - aux

    # --- 2. Nonlinear variables step ---
    calc_rhs_nl(rhs, Y_old, algs,pars,  t)
    for l in range(NL_START, NL_END):
        Y_new[l] = Y_old[l] + dt * rhs[l]
