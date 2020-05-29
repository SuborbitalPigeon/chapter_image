import cueimgproc
from matplotlib import figure

from .routines import histogram, plottools
import definitions


def plot_equalise_hist() -> figure.Figure:
    img = cueimgproc.GreyImage.open(definitions.PROJECT_ROOT / "data" / "stepped.tiff")

    fig, ax = plottools.create_subplots(0.3, ncols=3, sharex="all", sharey="all")

    histogram.plot_histogram(img.grey, ax[0])
    ax[0].set(title="Original image")

    global_equalised = img.apply_filter(cueimgproc.EqualiseFilter(False))
    histogram.plot_histogram(global_equalised.grey, ax[1])
    ax[1].set(title="Global equalisation")

    local_equalised = img.apply_filter(cueimgproc.EqualiseFilter(True))
    histogram.plot_histogram(local_equalised.grey, ax[2])
    ax[2].set(title="Adaptive equalisation")

    return fig
