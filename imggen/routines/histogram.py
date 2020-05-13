from matplotlib import axes
import numpy as np
from skimage import exposure


def plot_histogram(img: np.ndarray, ax: axes.Axes) -> None:
    # noinspection PyTypeChecker
    hist, bin_centres = exposure.histogram(img)
    ax.plot(bin_centres, hist)
    ax.set(xlim=(0.0, 1.0), label="Frequency")
    ax.set_ylim(bottom=0.0)
