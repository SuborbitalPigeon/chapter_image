from cueimgproc.nodes import image, preprocessing
from matplotlib import figure
from matplotlib import pyplot as plt

from .routines import histogram
import definitions


# noinspection PyTypeChecker
def plot_equalise_hist() -> figure.Figure:
    img = image.GreyImage.open(definitions.PROJECT_ROOT / "data" / "stepped.tiff")

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(7, 2.5),
        dpi=150,
        constrained_layout=True,
        sharex="all",
        sharey="all",
    )

    histogram.plot_histogram(img.grey, ax[0])
    ax[0].set(title="Original image")

    global_equalised = img.apply_filter(preprocessing.EqualiseFilter(False))
    histogram.plot_histogram(global_equalised.grey, ax[1])
    ax[1].set(title="Global equalisation")

    local_equalised = img.apply_filter(preprocessing.EqualiseFilter(True))
    histogram.plot_histogram(local_equalised.grey, ax[2])
    ax[2].set(title="Local equalisation")

    return fig
