import logging

from matplotlib import figure
import numpy as np
from skimage import io, transform

from .routines import plottools
import definitions


_log = logging.getLogger(__name__)


def plot_affine() -> figure.Figure:
    img = io.imread(definitions.PROJECT_ROOT / "data" / "monkey.jpg")
    affine = transform.AffineTransform(
        scale=(0.8, 0.6), rotation=np.pi / 12, shear=-np.pi / 6, translation=(50, 10)
    )

    _log.debug(affine)

    transformed = transform.warp(img, affine.inverse)

    fig, ax = plottools.create_subplots(0.3, ncols=2, sharey="all", frameon=False)
    plottools.remove_ticks(ax)
    ax[0].imshow(img)
    ax[1].imshow(transformed)

    return fig
