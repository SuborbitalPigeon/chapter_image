import time
from typing import Any, Callable

from matplotlib import axes, colors, figure
import numpy as np
from skimage import io, filters, restoration, util

from .routines import plottools

import definitions


def _load_image() -> np.ndarray:
    img = io.imread(definitions.DATA_DIR / "noisy.png")
    return util.img_as_float(img)


def _plot_denoise(
    ax: axes.Axes, func: Callable, title="Denoised", **kwargs: Any
) -> None:
    img = _load_image()

    start = time.process_time()
    denoised = func(img, **kwargs)
    end = time.process_time()

    ax.imshow(denoised, norm=colors.Normalize(0.0, 1.0))
    ax.set(title=f"{title}\nTime: {(end - start) * 1000.0:.0f} ms")


def _plot_original(ax: axes.Axes) -> None:
    data = _load_image()

    ax.imshow(data, norm=colors.Normalize(0.0, 1.0))
    ax.set(title="Original image")


def plot_denoise_all() -> figure.Figure:
    fig, axs = plottools.create_subplots(1.2, 3, 2, sharex="all")
    gs = axs[0][0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    ax_original = fig.add_subplot(gs[0, :])

    plottools.remove_ticks(ax_original)
    plottools.remove_ticks(axs)

    _plot_original(ax_original)
    _plot_denoise(axs[1, 0], filters.median, "Median filter")
    _plot_denoise(
        axs[1, 1], filters.gaussian, "Gaussian filter", multichannel=True, sigma=2.0
    )
    _plot_denoise(
        axs[2, 0],
        restoration.denoise_nl_means,
        "Non-local means, $h=0.02$",
        multichannel=True,
        h=0.03,
    )
    _plot_denoise(
        axs[2, 1],
        restoration.denoise_wavelet,
        "Wavelet",
        multichannel=True,
        convert2ycbcr=True,
        rescale_sigma=True,
    )

    return fig
