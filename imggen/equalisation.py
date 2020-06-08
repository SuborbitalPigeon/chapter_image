import cueimgproc
from matplotlib import figure

from .routines import plottools
import definitions


def plot_equalise() -> figure.Figure:
    img = cueimgproc.Image.open(definitions.PROJECT_ROOT / "data" / "stepped.tiff")
    global_equalised = img.apply_filter(cueimgproc.EqualiseFilter(False))
    local_equalised = img.apply_filter(cueimgproc.EqualiseFilter(True))

    fig, ax = plottools.create_subplots(0.85, 3, sharex="all")

    ax[0].set_title("Original image")
    plot = img.plot(ax[0])

    ax[1].set_title("After global equalisation")
    global_equalised.plot(ax[1])

    ax[2].set_title("After adaptive equalisation")
    local_equalised.plot(ax[2])

    cb = fig.colorbar(plot, ax=ax, fraction=0.2, location="bottom")
    cb.set_label("Scale common to all images")

    return fig
