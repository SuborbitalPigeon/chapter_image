import cueimgproc
from matplotlib import figure
import numpy as np

import definitions
from .routines import plottools


def plot_all_thresholds() -> figure.Figure:
    img = cueimgproc.GreyImage.open(definitions.DATA_DIR / "stepped.tiff")

    fig, ax = plottools.create_subplots(
        0.9, len(cueimgproc.ThresholdType) // 2, 2, sharex="all", sharey="all"
    )
    ax = ax.flatten()

    for i, t in enumerate(cueimgproc.ThresholdType):
        if t == cueimgproc.ThresholdType.LOCAL:
            filt = cueimgproc.ThresholdFilter(t, block_size=5)
            binary = img.apply_filter(filt)
        else:
            filt = cueimgproc.ThresholdFilter(t)
            binary = img.apply_filter(filt)

        if t.is_global:
            thresh_str = f"Threshold: {filt.threshold_value:.3f}"
        else:
            thresh_str = f"Average threshold: {np.mean(filt.threshold_value):.3f}"

        ax[i].set_title(f"{t.name.title()}\n{thresh_str}")

        binary.plot(ax[i], cmap="Set1")

    return fig
