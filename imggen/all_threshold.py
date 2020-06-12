import cueimgproc
from matplotlib import figure
import numpy as np

import definitions
from .routines import plottools


def plot_all_thresholds() -> figure.Figure:
    img = cueimgproc.Image.open(definitions.DATA_DIR / "stepped.tiff")

    # noinspection PyTypeChecker
    fig, ax = plottools.create_subplots(
        0.9, len(cueimgproc.ThresholdType) // 2, 2, sharex="all", sharey="all"
    )
    ax = ax.flatten()

    # noinspection PyTypeChecker
    for i, t in enumerate(cueimgproc.ThresholdType):
        if t == cueimgproc.ThresholdType.LOCAL:
            filter_ = cueimgproc.ThresholdFilter(t, block_size=5)
            binary = img.filter(filter_)
        else:
            filter_ = cueimgproc.ThresholdFilter(t)
            binary = img.filter(filter_)

        if t.is_global:
            thresh_str = f"Threshold: {filter_.threshold_value:.3f}"
        else:
            thresh_str = f"Average threshold: {np.mean(filter_.threshold_value):.3f}"

        ax[i].set_title(f"{t.name.title()}\n{thresh_str}")

        binary.plot(ax[i], cmap="binary")

    return fig
