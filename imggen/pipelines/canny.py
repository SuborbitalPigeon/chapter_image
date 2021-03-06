import cueimgproc
from matplotlib import colors, figure
import numpy as np
import seaborn as sns

import definitions
from ..routines import plottools


def _do_plotting(
    img: cueimgproc.Image, binary: cueimgproc.Image, regions: np.ndarray
) -> figure.Figure:
    fig, ax = plottools.create_subplots(0.6, 3, sharex="all")

    img.plot(ax[0])
    ax[0].set_title("Original image")

    binary.plot(ax[1])
    ax[1].set_title("Thresholded image")

    cmap = colors.ListedColormap(sns.husl_palette(len(np.unique(regions))).as_hex())
    bounds = np.asarray(np.unique(regions))
    norm = colors.BoundaryNorm(bounds, cmap.N)

    regions = np.ma.masked_equal(regions, 0, False)
    ax[2].imshow(regions, cmap=cmap, norm=norm)
    ax[2].set_title(f"Regions: {cmap.N}")

    return fig


def plot_threshold() -> figure.Figure:
    img = cueimgproc.Image.open(definitions.DATA_DIR / "stepped.tiff")
    binary = img.filter(cueimgproc.CannyFilter(), cueimgproc.BinaryFilter())
    labelled_image = cueimgproc.LabelledImage(binary.data, img)
    return _do_plotting(img, binary, labelled_image.region_map)


def plot_extra_threshold() -> figure.Figure:
    img = cueimgproc.Image.open(definitions.DATA_DIR / "stepped.tiff")
    binary = img.filter(
        cueimgproc.CannyFilter(),
        cueimgproc.BinaryFilter(),
        cueimgproc.RemoveSmallObjectsFilter(32),
    )

    labelled_image = cueimgproc.LabelledImage(binary.data, img)
    eccentricities = labelled_image.region_properties("eccentricity")
    idxs = eccentricities[eccentricities["eccentricity"] < 0.9].index.values

    return _do_plotting(img, binary, labelled_image.filter_regions(idxs))
