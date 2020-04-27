from cueimgproc.nodes import image, preprocessing
from matplotlib import figure
from matplotlib import pyplot as plt
from skimage import exposure

import definitions


# noinspection PyTypeChecker
def plot_equalise_hist() -> figure.Figure:
    img = image.GreyImage.open(definitions.PROJECT_ROOT / "data" / "stepped.tiff")

    fig, ax = plt.subplots(ncols=3, figsize=(7, 2.5), dpi=150, constrained_layout=True, sharex="all", sharey='all')

    hist, bin_centres = exposure.histogram(img.grey)
    ax[0].plot(bin_centres, hist)
    ax[0].set(xlim=(0.0, 1.0), ylabel="Frequency", title="Original image")

    global_equalised = img.apply_filter(preprocessing.EqualiseFilter(False))
    hist, bin_centres = exposure.histogram(global_equalised.grey)
    ax[1].plot(bin_centres, hist)
    ax[1].set(xlabel="Value", title="Global equalisation")

    local_equalised = img.apply_filter(preprocessing.EqualiseFilter(True))
    hist, bin_centres = exposure.histogram(local_equalised.grey)
    ax[2].plot(bin_centres, hist)
    ax[2].set(title="Local equalisation")

    return fig
