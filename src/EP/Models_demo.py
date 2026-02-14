import timeit
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.EP.ModelA import TTCellModelExt as modelA
from src.EP.ModelB import TTCellModelChannel as modelB
from src.EP.ModelC import TTCellModelFull as modelC

# Parameters for time configuration
ti, tf, dt, dtS = 1000, 1100, 0.01, 1
n=256
g=True
# Function to plot with uncertainty shades
# Function to plot with uncertainty shades + representative curves
def plot_with_shades(ax, time_points, waveforms, label, color, alpha=0.2, n_representative=5):
    waveforms = np.array(waveforms)
    min_waveform = np.min(waveforms, axis=0)
    max_waveform = np.max(waveforms, axis=0)
    mean_waveform = np.mean(waveforms, axis=0)

    # Mean waveform
    time_points=time_points[:len(mean_waveform)]
    ax.plot(time_points, mean_waveform, color=color, lw=2.5, label=label)

    # Uncertainty envelope
    ax.fill_between(time_points, min_waveform, max_waveform, color=color, alpha=alpha)

    # Plot a few representative individual solutions
    idxs = np.random.choice(len(waveforms), size=min(n_representative, len(waveforms)), replace=False)
    for i in idxs:
        ax.plot(time_points, waveforms[i], color=color, lw=0.8, alpha=0.6)



# Create the figure and axes for the subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Plot Model A on the first subplot
modelA.setSizeParameters(ti, tf, dt, dtS)
tp = modelA.getEvalPoints()

distA = modelA.getDist(low=0.5, high=1.5)
samplesA = distA.sample(n, rule="latin_hypercube").T
resultsA = modelA.run(samplesA)

waveformsA = [res['Wf'] for res in resultsA]
plot_with_shades(axes[0], tp, waveformsA, "Model A", color="blue")

# Plot Model B (disease vs healthy) on the second subplot
modelB.setSizeParameters(ti, tf, dt, dtS)
distB_healthy = modelB.getDist(low=0.5, high=1.5, disse_h=0.00000001, disse_l=0.)
distB_disease = modelB.getDist(low=0.5, high=1.5, disse_h=1.0, disse_l=.99)

# Healthy model results
samplesB_healthy = distB_healthy.sample(n, rule="latin_hypercube").T
resultsB_healthy = modelB.run(samplesB_healthy)
waveformsB_healthy = [res['Wf'] for res in resultsB_healthy]
plot_with_shades(axes[1], tp, waveformsB_healthy, "Model B Healthy", color="#8B0000", alpha=0.3)

# Disease model results
samplesB_disease = distB_disease.sample(n, rule="latin_hypercube").T
resultsB_disease = modelB.run(samplesB_disease, use_gpu=g, regen=True)
waveformsB_disease = [res['Wf'] for res in resultsB_disease]
plot_with_shades(axes[1], tp, waveformsB_disease, "Model B Disease", color="#FF6F61", alpha=0.3)


# Plot Model B (full parameter space) on the third subplot
distB_full = modelB.getDist(low=0.5, high=1.5, disse_l=0, disse_h=1)  # Full parameter space

samplesB_full = distB_full.sample(n, rule="latin_hypercube").T
resultsB_full = modelB.run(samplesB_full, use_gpu=g, regen=True)
waveformsB_full = [res['Wf'] for res in resultsB_full]
plot_with_shades(axes[2], tp, waveformsB_full, "Model B", color="red", alpha=0.3)

axes[0].set_ylabel("Waveform Amplitude (mV)", fontsize=24)
axes[0].legend(loc="upper right", fontsize=24)
axes[1].legend(loc="upper right", fontsize=24)
axes[2].legend(loc="upper right", fontsize=24)
axes[0].set_xticks([])  # Remove x-ticks for this subplot
axes[1].set_xticks([])  # Remove x-ticks for this subplot
axes[2].set_xticks([])  # Remove x-ticks for this subplot

# Add overall title and adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0., wspace=0.)  # Reduce vertical and horizontal space

# Save and show the plot
plt.savefig("model_comparison_with_shades.png", bbox_inches="tight")



color_waveform = "green"             # color for the plot


# -------------------------------
# Set simulation size and get evaluation points
modelC.setSizeParameters(ti, tf, dt, dtS)
time_points = modelC.getEvalPoints()

# -------------------------------
# Create parameter distribution: 12 parameters in [0.5, 1.5]
distC = modelC.getDist(low=0.999, high=0.9999)
samplesC = distC.sample(n, rule="latin_hypercube").T

# Run model
resultsC = modelC.run(samplesC, use_gpu=True)  # set use_gpu=False if you want CPU

# Extract waveforms
waveformsC = [res['Wf'] for res in resultsC]

# -------------------------------
# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
plot_with_shades(ax, time_points, waveformsC, "Model C", color=color_waveform)

ax.set_xlabel("Time (ms)", fontsize=14)
ax.set_ylabel("Waveform Amplitude (mV)", fontsize=14)
ax.legend(loc="upper right", fontsize=12)
ax.set_title("Model C - Parameter Sweep [0.5, 1.5]", fontsize=16)

plt.tight_layout()
plt.savefig("modelC_waveforms.png", bbox_inches="tight")
plt.show()