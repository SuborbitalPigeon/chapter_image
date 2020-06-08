from typing import Optional, Tuple, Union

from matplotlib import axes, figure
from matplotlib import pyplot as plt
import numpy as np

import definitions


def create_subplots(
    height_ratio: float, nrows: Optional[int] = 1, ncols: Optional[int] = 1, **kwargs,
) -> Tuple[figure.Figure, Union[axes.Axes, np.ndarray]]:
    """Create a figure with axes, using a common style.

    Args:
        height_ratio: The ratio of height to width.
        nrows: The number of rows of axes to create.
        ncols: The number of columns of axes to create.
        **kwargs: Additional arguments to pass to :func:`~matplotlib.pyplot.subplots`.

    Returns:
        fig: The figure object.
        axes: Either the single axes object, or an array of them.
    """
    size_inches = (
        definitions.PAGE_WIDTH / 2.54,
        (definitions.PAGE_WIDTH * height_ratio) / 2.54,
    )
    fig, ax = plt.subplots(
        nrows, ncols, figsize=size_inches, dpi=150, constrained_layout=True, **kwargs
    )
    return fig, ax


def remove_ticks(ax: Union[axes.Axes, np.ndarray]) -> None:
    """Remove ticks from an Axes or group of axes.

    Args:
        ax: Axis or group of them
    """
    if isinstance(ax, np.ndarray):
        for axis in ax.flatten():
            remove_ticks(axis)
    else:
        ax.axis("off")
