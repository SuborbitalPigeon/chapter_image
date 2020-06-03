import cueimgproc
from matplotlib import cm, colors, figure
import numpy as np

import definitions
from ..routines import plottools


def _do_plotting(img: cueimgproc.GreyImage, regions: np.ndarray) -> figure.Figure:
    fig, ax = plottools.create_subplots(0.6, 3, sharex="all")

    img.plot(ax[0])
    ax[0].set_title("Original image")

    ax[1].imshow(regions > 0.5, cmap="binary")
    ax[1].set_title("Thresholded image")

    cmap = cm.get_cmap("twilight", len(np.unique(regions)))
    bounds = np.asarray(np.unique(regions.data))
    norm = colors.BoundaryNorm(bounds, cmap.N)

    region_plot = ax[2].imshow(regions, cmap=cmap, norm=norm)
    ax[2].set_title(f"Regions: {cmap.N}")
    fig.colorbar(region_plot, ax=ax[2], location="bottom")

    return fig


def plot_threshold() -> figure.Figure:
    img = cueimgproc.GreyImage.open(definitions.DATA_DIR / "stepped.tiff")
    binary = img.apply_filter(
        cueimgproc.ThresholdFilter(cueimgproc.ThresholdType.SAUVOLA)
    )

    regions = cueimgproc.label_image_with_props(binary.grey)[0]
    return _do_plotting(img, regions)


def plot_extra_threshold() -> figure.Figure:
    img = cueimgproc.GreyImage.open(definitions.DATA_DIR / "stepped.tiff")
    binary = img.apply_filters(
        [
            cueimgproc.ThresholdFilter(cueimgproc.ThresholdType.SAUVOLA),
            cueimgproc.RemoveSmallObjectsFilter(32),
        ]
    )

    regions, df = cueimgproc.label_image_with_props(binary.grey, 0.9)

    return _do_plotting(img, regions)
