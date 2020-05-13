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
        scale=(1.3, 1.5), rotation=np.pi / 6, shear=-np.pi / 6, translation=(-50, -300)
    )

    _log.debug(affine)

    transformed = transform.warp(img, affine)

    fig, ax = plottools.create_subplots((5.5, 2), ncols=2)
    ax[0].imshow(img)
    ax[1].imshow(transformed)

    return fig
