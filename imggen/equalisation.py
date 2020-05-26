import cueimgproc
from matplotlib import figure

from .routines import plottools
import definitions


def plot_equalise() -> figure.Figure:
    img = cueimgproc.GreyImage.open(definitions.PROJECT_ROOT / "data" / "stepped.tiff")
    global_equalised = img.apply_filter(cueimgproc.EqualiseFilter(False))
    local_equalised = img.apply_filter(cueimgproc.EqualiseFilter(True))

    fig, ax = plottools.create_subplots((5, 4), 3)

    ax[0].set_title("Original image")
    plot = img.plot(ax[0])

    ax[1].set_title("After global equalisation")
    global_equalised.plot(ax[1])

    ax[2].set_title("After local equalisation")
    local_equalised.plot(ax[2])

    cb = fig.colorbar(plot, ax=ax, shrink=0.8, location="bottom")
    cb.set_label("Scale common to all images")

    return fig
