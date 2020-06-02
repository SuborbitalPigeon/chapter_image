import cueimgproc
from matplotlib import cm, colors, figure
from skimage import io

import definitions

from .routines import plottools


def plot_wavelet_decompose() -> figure.Figure:
    img = io.imread(definitions.DATA_DIR / "monkey.tiff")
    img = cueimgproc.GreyImage.from_colour(img)

    decompose = cueimgproc.WaveletDecompose("db1", 3)
    decompose(img)

    fig, axs = plottools.create_subplots(1.5, 4, 3, sharex='all', sharey='all')
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:  # Remove the top row and replace with one wide plot
        ax.remove()
    ax_approx = fig.add_subplot(gs[0, :])
    plottools.remove_ticks(ax_approx)

    approx = ax_approx.imshow(decompose.approximation)
    ax_approx.set_title("Approximation")
    fig.colorbar(approx, ax=ax_approx, location="bottom", fraction=0.02)

    plottools.remove_ticks(axs)

    directions = ['horizontal', 'vertical', 'diagonal']
    for i in range(1, 4):
        level_coeffs = decompose.get_level_coeffs(i)
        vmin = min([direction.min() for direction in level_coeffs])
        vmax = max([direction.max() for direction in level_coeffs])
        norm = colors.Normalize(vmin, vmax)

        for j in range(3):
            axs[i, j].imshow(level_coeffs[j], norm=norm)
            axs[i, j].set_title(f"Level {i}, {directions[j]}")

        fig.colorbar(
            cm.ScalarMappable(norm), ax=axs[i, :], location="bottom", fraction=0.02
        )

    return fig
