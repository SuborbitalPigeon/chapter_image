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
    denoised = func(img, **kwargs)

    ax.imshow(denoised, norm=colors.Normalize(0.0, 1.0))
    ax.set(title=title)


def _plot_original(ax: axes.Axes) -> None:
    data = _load_image()

    ax.imshow(data, norm=colors.Normalize(0.0, 1.0))
    ax.set(title="Original image")


def plot_denoise_median() -> figure.Figure:
    fig, axs = plottools.create_subplots(0.4, ncols=2, sharey="all")
    plottools.remove_ticks(axs)

    _plot_original(axs[0])

    selem = np.ones((7, 7, 3))
    _plot_denoise(axs[1], filters.median, selem=selem)

    return fig


def plot_denoise_gauss() -> figure.Figure:
    fig, axs = plottools.create_subplots(0.4, ncols=2, sharey="all")
    plottools.remove_ticks(axs)

    _plot_original(axs[0])
    _plot_denoise(axs[1], filters.gaussian, sigma=2, multichannel=True)

    return fig


def plot_denoise_nlmeans() -> figure.Figure:
    fig, axs = plottools.create_subplots(0.4, ncols=2, sharey="all")
    plottools.remove_ticks(axs)

    _plot_original(axs[0])
    _plot_denoise(axs[1], restoration.denoise_nl_means, multichannel=True, h=0.02)

    return fig


def plot_denoise_wavelet() -> figure.Figure:
    fig, axs = plottools.create_subplots(0.4, ncols=2, sharey="all")
    plottools.remove_ticks(axs)

    _plot_original(axs[0])
    _plot_denoise(
        axs[1],
        restoration.denoise_wavelet,
        convert2ycbcr=True,
        multichannel=True,
        rescale_sigma=True,
    )

    return fig


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
    _plot_denoise(axs[1, 1], filters.gaussian, "Gaussian filter", multichannel=True)
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
