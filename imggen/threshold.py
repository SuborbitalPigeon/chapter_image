from __future__ import annotations

from typing import TYPE_CHECKING, Union

from cueimgproc.nodes import common, image, threshold
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib import figure
    import numpy as np

from .routines import histogram
import definitions


def _do_plotting(
    fig: figure.Figure, ax: np.ndarray, threshold_type: threshold.ThresholdType
) -> Union[float, np.ndarray]:
    img = image.GreyImage.open(definitions.DATA_DIR / "stepped.tiff")
    img = img.apply_filter(common.RemoveAlphaFilter())

    filter = threshold.ThresholdFilter(threshold_type)
    threshed = img.apply_filter(filter)

    original_plot = img.plot(ax[0])
    ax[0].set(title="Original image")
    fig.colorbar(original_plot, ax=ax[0])

    thresh_plot = threshed.plot(ax[1], {"cmap": plt.cm.get_cmap("viridis", 2)})
    ax[1].set(title=f"Image after {threshold_type.name.lower()} thresholding")
    fig.colorbar(thresh_plot, ax=ax[1])

    histogram.plot_histogram(img.grey, ax[2])
    ax[2].set(xlabel="Value", ylabel="Frequency")

    return filter.threshold_value


def plot_global_threshold() -> figure.Figure:
    fig, ax = plt.subplots(3, figsize=(6, 4), dpi=150, constrained_layout=True)

    value = _do_plotting(fig, ax, threshold.ThresholdType.OTSU)
    ax[2].axvline(value, color="red")

    return fig


def plot_local_threshold() -> figure.Figure:
    fig, ax = plt.subplots(4, figsize=(6, 5), dpi=150, constrained_layout=True)

    value = _do_plotting(fig, ax, threshold.ThresholdType.SAUVOLA)
    ax[2].hist(value.ravel(), bins="auto", color="red", alpha=0.5)
    ax[2].set(xlabel="Value", ylabel="Frequency")

    value_image = image.GreyImage(value)
    value_image.plot(ax[3])
    ax[3].set(title="Threshold value")

    return fig