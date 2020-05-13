from typing import Optional, Tuple, Union

from matplotlib import axes, figure
from matplotlib import pyplot as plt
import numpy as np


def create_subplots(
    figsize: Tuple[float, float],
    nrows: Optional[int] = 1,
    ncols: Optional[int] = 1,
    **kwargs,
) -> Tuple[figure.Figure, Union[axes.Axes, np.ndarray]]:
    fig, ax = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=150, constrained_layout=True, **kwargs
    )
    return fig, ax
