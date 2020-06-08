import cueimgproc
from matplotlib import colors, figure
import numpy as np

from .routines import plottools
import definitions


def plot_sobel() -> figure.Figure:
    img = cueimgproc.Image.open(definitions.PROJECT_ROOT / "data" / "spiral.png")

    edges = cueimgproc.DirectionalEdgeDetector(cueimgproc.EdgeFilterType.SOBEL)
    edges(img)

    amplitude = edges.edge_magnitude
    direction = edges.edge_direction * (180.0 / np.pi)

    fig, axs = plottools.create_subplots(0.5, 2, 3)
    plottools.remove_ticks(axs)

    spiral_plot = img.plot(axs[0, 0], cmap="Greys")
    axs[0, 0].set(title="Image")
    fig.colorbar(spiral_plot, ax=axs[0, 0])

    h_plot = axs[0, 1].imshow(edges.horizontal_edges, cmap="coolwarm")
    axs[0, 1].set(title="Horizontal edges")
    fig.colorbar(h_plot, ax=axs[0, 1])

    h_plot = axs[0, 2].imshow(edges.vertical_edges, cmap="coolwarm")
    axs[0, 2].set(title="Vertical edges")
    fig.colorbar(h_plot, ax=axs[0, 2])

    amp_plot = axs[1, 1].imshow(amplitude, cmap="Greys")
    axs[1, 1].set(title="Magnitude")
    fig.colorbar(amp_plot, ax=axs[1, 1])

    dir_plot = axs[1, 2].imshow(
        direction,
        cmap="coolwarm",
        interpolation="nearest",
        norm=colors.Normalize(-180.0, 180.0),
    )
    axs[1, 2].set(title="Direction")
    dir_cb = fig.colorbar(dir_plot, ax=axs[1, 2])
    dir_cb.set_label("Degrees")

    fig.delaxes(axs[1, 0])

    return fig
