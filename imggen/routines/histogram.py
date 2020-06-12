from matplotlib import axes
import numpy as np
import seaborn as sns


def plot_histogram(img: np.ndarray, ax: axes.Axes) -> None:
    """Find a histogram for an image, and plot it.

    Args:
        img: The image to find the histogram for.
        ax: The Axes to place the histogram graph into.
    """
    sns.kdeplot(img.flatten(), ax=ax, label="Pixel value")
    ax.set(xlim=(0.0, 1.0), xlabel="Value", ylabel="Relative frequency")
    ax.set_ylim(0.0)
