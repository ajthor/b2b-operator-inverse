"""
Test script for Chladni publication plot gridspec layout.

Tests the 2x3 + 2x3 grid layout with shared colorbars using placeholder data.
This allows us to verify the layout fits on screen before integrating with models.

To run: python test_chladni_gridspec.py
"""

import numpy as np
import matplotlib.pyplot as plt


# Set publication-quality rcParams
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 1.0,
    }
)
# plt.rcParams["figure.constrained_layout.use"] = True


# Create random imshow data
def create_placeholder_data():
    data = np.random.rand(100, 100)
    return data


# Set up the figure with single unified gridspec
fig = plt.figure(figsize=(6.5, 2.0), layout="constrained")
fig.set_constrained_layout_pads(
    w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
)

# Single gridspec: 2 rows x 8 columns
# Columns: [plot, plot, plot, colorbar, plot, plot, plot, colorbar]
gs = fig.add_gridspec(
    2,
    8,
    width_ratios=[1, 1, 1, 0.05, 1, 1, 1, 0.05],
    hspace=0.0,
    wspace=0.0,
    left=0,
    right=1,
    top=1,
    bottom=0,
)

# Create parent axes for shared labels (invisible, just for labels)
ax_left_parent = fig.add_subplot(gs[:, 0:3], frameon=False)
ax_left_parent.tick_params(
    labelcolor="none", top=False, bottom=False, left=False, right=False
)
ax_left_parent.set_xlabel("x", labelpad=-8)
ax_left_parent.set_ylabel("y", labelpad=-8)
ax_left_parent.set_title("Chladni Patterns for Various Models")

ax_right_parent = fig.add_subplot(gs[:, 4:7], frameon=False)
ax_right_parent.tick_params(
    labelcolor="none", top=False, bottom=False, left=False, right=False
)
ax_right_parent.set_xlabel("x", labelpad=-8)
ax_right_parent.set_ylabel("y", labelpad=-8)
ax_right_parent.set_title("Chladni Patterns for Various Models")

# Model names for annotation (placeholder)
model_names = ["Model A", "Model B", "Model C", "Model D", "Model E", "Model F"]

# Plot left grid (columns 0-2)
for i in range(2):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        im = ax.imshow(create_placeholder_data(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # Add annotation with white text on dark background
        model_idx = i * 3 + j
        ax.text(
            0.05, 0.95, model_names[model_idx],
            transform=ax.transAxes,
            fontsize=6,
            color='white',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

# Shared colorbar for left grid (column 3)
cax_left = fig.add_subplot(gs[:, 3])
cbar_left = fig.colorbar(im, cax=cax_left, use_gridspec=True)
cbar_left.set_label("Intensity")

# Plot right grid (columns 4-6)
for i in range(2):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j + 4])
        im = ax.imshow(create_placeholder_data(), cmap="plasma", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # Add annotation with white text on dark background
        model_idx = i * 3 + j
        ax.text(
            0.05, 0.95, model_names[model_idx],
            transform=ax.transAxes,
            fontsize=6,
            color='white',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

# Shared colorbar for right grid (column 7)
cax_right = fig.add_subplot(gs[:, 7])
cbar_right = fig.colorbar(im, cax=cax_right, use_gridspec=True)
cbar_right.set_label("Intensity")

plt.savefig("test_chladni_gridspec.png", dpi=300)
