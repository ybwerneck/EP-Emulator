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

phase = 20000
ti, tf, dt, dtS = phase+000, phase+400, 0.01, 0.01
n = 256
use_gpu = True


# -------------------------------
# Plotting function

def plot_with_shades_and_apd(ax, time_points, waveforms, apds, label, color,
                             alpha_outer=0.20, alpha_inner=0.35,
                             n_representative=5):

    waveforms = np.array(waveforms)
    apds = np.array(apds)

    # -------------------------------
    # Waveform percentiles

    p5  = np.percentile(waveforms, 10, axis=0)
    p25 = np.percentile(waveforms, 30, axis=0)
    p50 = np.percentile(waveforms, 50, axis=0)
    p75 = np.percentile(waveforms, 70, axis=0)
    p95 = np.percentile(waveforms, 90, axis=0)

    time_points = time_points[:len(p50)]
    time_points = [t - phase for t in time_points]

    # -------------------------------
    # Outer uncertainty band (10–90)

    ax.fill_between(
        time_points,
        p5,
        p95,
        color=color,
        alpha=alpha_outer
    )

    # -------------------------------
    # Inner uncertainty band (30–70)

    ax.fill_between(
        time_points,
        p25,
        p75,
        color=color,
        alpha=alpha_inner
    )

    # -------------------------------
    # Select samples inside the outer band

    inside_band = np.all((waveforms >= p5) & (waveforms <= p95), axis=1)

    waveforms_band = waveforms[inside_band]
    apds_band = apds[inside_band]

    if len(apds_band) == 0:
        apds_band = apds

    # -------------------------------
    # Representative trajectories

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
            alpha=0.35
        )

    # -------------------------------
    # Trajectory closest to the median

    dist = np.linalg.norm(waveforms - p50, axis=1)
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
    # APDs for the representative trajectory

    rep_apds = apds[rep_idx]

    labels = ["APD30", "APD50", "APD80"][:len(rep_apds)]

    base = np.array(mcolors.to_rgb(color))
    def scale_color(base, factor):
        base = np.array(base)
        if factor < 1:
            return base * factor
        else:
            return base + (1 - base) * (factor - 1)

    factors = [0.4, 1.0, 1.4]
    colors = [scale_color(base, f) for f in factors]

    # APD bounds computed only from the outer band
    apd_min = np.min(apds_band, axis=0)
    apd_max = np.max(apds_band, axis=0)

    for i, apd_time in enumerate(rep_apds):

        c = colors[i]

        y = np.interp(apd_time, time_points, rep_waveform)

        ax.scatter(
            apd_time,
            y,
            color=c,
            edgecolor="black",
            zorder=5,
            label=f"{labels[i]}"
        )

        ax.errorbar(
            apd_time,
            y,
            xerr=[[apd_time - apd_min[i]], [apd_max[i] - apd_time]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            alpha=0.9
        )

    # -------------------------------
    # Additional QoIs

    v_rest = rep_waveform[0]
    t_rest = time_points[0]

    peak_idx = np.argmax(rep_waveform)
    v_peak = rep_waveform[peak_idx]
    t_peak = time_points[peak_idx]

    dv = np.gradient(rep_waveform, time_points)
    dv_idx = np.argmax(dv)

    t_dv = time_points[dv_idx]
    v_dv = rep_waveform[dv_idx]

    ax.scatter(t_rest, v_rest, marker="s", color="black", s=40, label="V_rest")
    ax.scatter(t_peak, v_peak, marker="s", color="black", s=40, label="V_peak")
    ax.scatter(t_dv, v_dv, marker="D", color="red", s=45, label="Max dV/dt")


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
axes[0].set_ylabel("Membrane Potential (mV)", fontsize=14)


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
# Final plot style

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