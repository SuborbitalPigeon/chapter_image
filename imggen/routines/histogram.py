from __future__ import annotations

from typing import TYPE_CHECKING

from skimage import exposure

if TYPE_CHECKING:
    from matplotlib import axes
    import numpy as np


def plot_histogram(img: np.ndarray, ax: axes.Axes) -> None:
    hist, bin_centres = exposure.histogram(img)
    ax.plot(bin_centres, hist)
    ax.set(xlim=(0.0, 1.0), label="Frequency")
    ax.set_ylim(bottom=0.0)