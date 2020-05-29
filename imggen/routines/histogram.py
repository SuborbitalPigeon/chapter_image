from matplotlib import axes
import numpy as np
from skimage import exposure


def plot_histogram(img: np.ndarray, ax: axes.Axes, nbins: int = 50) -> None:
    """Find a histogram for an image, and plot it.

    Args:
        img: The image to find the histogram for.
        ax: The Axes to place the histogram graph into.
        nbins: The number of bins to use.
    """
    # noinspection PyTypeChecker
    hist, bin_centres = exposure.histogram(img, nbins)
    container = ax.stem(bin_centres, hist, markerfmt=".", use_line_collection=True)
    ax.set(xlim=(0.0, 1.0), xlabel="Value", ylabel="Frequency")
    ax.set_ylim(0.0)

    container.stemlines.linewidth = 0.5
