from __future__ import annotations

from typing import Union

import cueimgproc
from matplotlib import figure
import numpy as np

from .routines import histogram, plottools
import definitions


def _do_plotting(
    fig: figure.Figure, ax: np.ndarray, threshold_type: cueimgproc.ThresholdType
) -> Union[float, np.ndarray]:
    img = cueimgproc.Image.open(definitions.DATA_DIR / "stepped.tiff")
    img = img.apply_filter(cueimgproc.RemoveAlphaFilter())

    filter = cueimgproc.ThresholdFilter(threshold_type)
    threshed = img.apply_filter(filter)

    original_plot = img.plot(ax[0])
    ax[0].set(title="Original image")
    fig.colorbar(original_plot, ax=ax[0])

    thresh_plot = threshed.plot(ax[1], cmap="binary")
    ax[1].set(title=f"Image after {threshold_type.name.capitalize()} thresholding")
    fig.colorbar(thresh_plot, ax=ax[1])

    histogram.plot_histogram(img.data, ax[2])

    return filter.threshold_value


def plot_global_threshold() -> figure.Figure:
    fig, ax = plottools.create_subplots(0.7, 3)

    value = _do_plotting(fig, ax, cueimgproc.ThresholdType.OTSU)
    ax[2].axvline(value, color="red")

    return fig


def plot_adaptive_threshold() -> figure.Figure:
    fig, ax = plottools.create_subplots(1, 4)

    value = _do_plotting(fig, ax, cueimgproc.ThresholdType.SAUVOLA)
    ax[2].hist(value.ravel(), bins="auto", color="red", alpha=0.5)
    ax[2].set(xlabel="Value", ylabel="Frequency")

    value_image = cueimgproc.Image(value)
    value_plot = value_image.plot(ax[3])
    ax[3].set(title="Threshold value")
    fig.colorbar(value_plot, ax=ax[3])

    return fig
