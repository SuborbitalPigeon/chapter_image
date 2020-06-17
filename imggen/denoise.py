import time
from typing import Any

import cueimgproc
from matplotlib import axes, colors, figure

from .routines import plottools

import definitions


def _plot_denoise(
    ax: axes.Axes, denoise_type: cueimgproc.DenoiseType, title: str, **kwargs: Any
) -> None:
    img = cueimgproc.Image.open(definitions.DATA_DIR / "noisy.png")

    denoiser = cueimgproc.DenoiseFilter(denoise_type, **kwargs)

    start = time.process_time()
    denoised = img.filter(denoiser)
    end = time.process_time()

    denoised.plot(ax, norm=colors.Normalize(0.0, 1.0))
    ax.set(title=f"{title}\nTime: {(end - start) * 1000.0:.0f} ms")


def _plot_original(ax: axes.Axes) -> None:
    data = cueimgproc.Image.open(definitions.DATA_DIR / "noisy.png")

    data.plot(ax, norm=colors.Normalize(0.0, 1.0))
    ax.set(title="Original image")


def plot_denoise_all() -> figure.Figure:
    fig, axs = plottools.create_subplots(1, 3, 3, sharex="all", sharey="all")
    gs = axs[0][0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    ax_original = fig.add_subplot(gs[0, :])

    plottools.remove_ticks(ax_original)
    plottools.remove_ticks(axs)

    _plot_original(ax_original)

    _plot_denoise(axs[1, 0], cueimgproc.DenoiseType.BILATERAL, "Bilateral")
    _plot_denoise(axs[1, 1], cueimgproc.DenoiseType.MEDIAN, "Median")
    _plot_denoise(axs[1, 2], cueimgproc.DenoiseType.NL_MEANS, "Non-local means", h=0.05)
    _plot_denoise(axs[2, 0], cueimgproc.DenoiseType.TV_BREGMAN, "TV: Bregman", weight=1)
    _plot_denoise(axs[2, 1], cueimgproc.DenoiseType.TV_CHAMBOLLE, "TV: Chambolle")
    _plot_denoise(axs[2, 2], cueimgproc.DenoiseType.WAVELET, "Wavelet")

    return fig
