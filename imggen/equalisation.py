from cueimgproc.nodes import image, preprocessing
from matplotlib import figure
from matplotlib import pyplot as plt

import definitions


def plot_equalise() -> figure.Figure:
    img = image.GreyImage.open(definitions.PROJECT_ROOT / "data" / "stepped.tiff")
    global_equalised = img.apply_filter(preprocessing.EqualiseFilter(False))
    local_equalised = img.apply_filter(preprocessing.EqualiseFilter(True))

    fig, ax = plt.subplots(3, figsize=(5, 4), dpi=150, constrained_layout=True)

    ax[0].set_title("Original image")
    plot = img.plot(ax[0])

    ax[1].set_title("After global equalisation")
    global_equalised.plot(ax[1])

    ax[2].set_title("After local equalisation")
    local_equalised.plot(ax[2])

    cb = fig.colorbar(plot, ax=ax, shrink=0.8, location="bottom")
    cb.set_label("Scale common to all images")

    return fig
