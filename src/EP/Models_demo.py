import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import sys, os
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.EP.ModelA import TTCellModelExt as modelA
from src.EP.ModelC import TTCellModelFull as modelC


# -------------------------------
# Simulation configuration
phase=1000
ti, tf, dt, dtS = phase+000, phase+400, 0.01, 0.01
n = 256
use_gpu = True


# -------------------------------
# Plotting function
labels=["APD30", "APD50", "APD80"]
import numpy as np
import matplotlib.colors as mcolors

def plot_with_shades_and_apd(ax, time_points, waveforms, apds, label, color,
                             alpha=0.25, n_representative=5):

    waveforms = np.array(waveforms)

    min_waveform = np.min(waveforms, axis=0)
    max_waveform = np.max(waveforms, axis=0)
    mean_waveform = np.mean(waveforms, axis=0)

    time_points = time_points[:len(mean_waveform)]
    time_points=[t-phase for t in time_points]
    # -------------------------------
    # Uncertainty band
    ax.fill_between(
        time_points,
        min_waveform,
        max_waveform,
        color=color,
        alpha=alpha
    )

    # -------------------------------
    # faint representative trajectories
    idxs = np.random.choice(
        len(waveforms),
        size=min(n_representative, len(waveforms)),
        replace=False
    )

    for i in idxs:
        ax.plot(
            time_points,
            waveforms[i],
            color=color,
            lw=0.4,
            alpha=0.4
        )

    # -------------------------------
    # choose representative trajectory
    dist = np.linalg.norm(waveforms - mean_waveform, axis=1)
    rep_idx = np.argmin(dist)

    rep_waveform = waveforms[rep_idx]

    ax.plot(
        time_points,
        rep_waveform,
        color=color,
        lw=2,
        label=f"{label}"
    )

    # -------------------------------
    # APD markers for representative trajectory

    apds = np.array(apds)

    if apds.ndim == 1:
        apds = apds[:, None]

    rep_apds = apds[rep_idx]

    labels = ["APD30", "APD50", "APD80"][:len(rep_apds)]

    base = np.array(mcolors.to_rgb(color))
    factors = np.linspace(0.7, 1.2, len(rep_apds))
    colors = [np.clip(base * f, 0, 1) for f in factors]
    # compute global APD ranges
    apd_min = np.min(apds, axis=0)
    apd_max = np.max(apds, axis=0)

    for i, apd_time in enumerate(rep_apds):

        c = colors[i]

        y = np.interp(apd_time, time_points, rep_waveform)

        # representative APD point
        ax.scatter(
            apd_time,
            y,
            color=c,
            edgecolor="black",
            zorder=5,
            label=f"{labels[i]}"
        )

        # small horizontal error bar showing range
        ax.errorbar(
            apd_time,
            y,
            xerr=[[apd_time - apd_min[i]], [apd_max[i] - apd_time]],
            fmt="none",
            ecolor=c,
            elinewidth=1,
            capsize=3,
            alpha=0.9
        )

            # -------------------------------
    # Additional QoIs for representative trajectory

    # V_rest (first value)
    v_rest = rep_waveform[0]
    t_rest = time_points[0]

    # V_peak
    peak_idx = np.argmax(rep_waveform)
    v_peak = rep_waveform[peak_idx]
    t_peak = time_points[peak_idx]

    # dV/dt max
    dv = np.gradient(rep_waveform, time_points)
    dv_idx = np.argmax(dv)

    dv_max = dv[dv_idx]
    t_dv = time_points[dv_idx]
    v_dv = rep_waveform[dv_idx]

    # plot V_rest
    ax.scatter(
        t_rest,
        v_rest,
        marker="s",
        color="black",
        s=40,
        label=f"V_rest"
    )

    # plot V_peak
    ax.scatter(
        t_peak,
        v_peak,
        marker="s",
        color="black",
        s=40,
        label=f"V_peak"
    )

    # plot dV/dt max
    ax.scatter(
        t_dv,
        v_dv,
        marker="D",
        color="red",
        s=45,
        label=f"dV/dt max"
    )
    
# -------------------------------
# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


# =====================================================
# EP_A (Model A)
# =====================================================

modelA.setSizeParameters(ti, tf, dt, dtS)

tpA = modelA.getEvalPoints()

distA = modelA.getDist(low=0.5, high=1.5)

samplesA = distA.sample(n, rule="latin_hypercube").T

resultsA = modelA.run(samplesA)

waveformsA = [res['Wf'] for res in resultsA]

apdsA = [tuple([res['APD30'], res['APD50'], res['APD80']]) for res in resultsA]

plot_with_shades_and_apd(
    axes[0],
    tpA,
    waveformsA,
    apdsA,
    label="EP_A",
    color="blue"
)

axes[0].set_title("EP_A (Ischemic Model)", fontsize=16)
axes[0].set_ylabel("Voltage (mV)", fontsize=14)


# =====================================================
# EP_B (Model C)
# =====================================================

modelC.setSizeParameters(ti, tf, dt, dtS)

tpC = modelC.getEvalPoints()

distC = modelC.getDist(low=0.75, high=1.25)

samplesC = distC.sample(n, rule="latin_hypercube").T

resultsC = modelC.run(samplesC, use_gpu=use_gpu)

waveformsC = [res['Wf'] for res in resultsC]

apdsC = [tuple([res['APD30'], res['APD50'], res['APD80']]) for res in resultsC]

print(apdsC)

plot_with_shades_and_apd(
    axes[1],
    tpC,
    waveformsC,
    apdsC,
    label="EP_B",
    color="green"
)

axes[1].set_title("EP_B (Fully Parameterized Model)", fontsize=16)


# -------------------------------
# Final styling

for ax in axes:
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylim(-90, 45)
    ax.legend(fontsize=11)

plt.tight_layout()

plt.savefig(
    "EP_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()