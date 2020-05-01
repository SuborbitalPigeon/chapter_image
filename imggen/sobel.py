from cueimgproc.nodes import edge, image
from matplotlib import colors, figure
from matplotlib import pyplot as plt
import numpy as np

import definitions


def plot_sobel() -> figure.Figure:
    img = image.GreyImage.open(definitions.PROJECT_ROOT / "data" / "spiral.png")

    edges = edge.DirectionalEdgeDetector(edge.EdgeFilterType.SOBEL)
    edges(img)

    amplitude = edges.edge_magnitude
    direction = edges.edge_direction * (180.0 / np.pi)

    fig, axs = plt.subplots(2, 3, dpi=150, figsize=(9, 5), constrained_layout=True)

    spiral_plot = img.plot(axs[0, 0], {"cmap": "Greys"})
    axs[0, 0].set(title="Image")
    fig.colorbar(spiral_plot, ax=axs[0, 0])

    h_plot = axs[0, 1].imshow(edges.horizontal_edges, cmap="coolwarm")
    axs[0, 1].set(title="Horizontal edges")
    fig.colorbar(h_plot, ax=axs[0, 1])

    h_plot = axs[0, 2].imshow(edges.vertical_edges, cmap="coolwarm")
    axs[0, 2].set(title="Vertical edges")
    fig.colorbar(h_plot, ax=axs[0, 2])

    amp_plot = axs[1, 1].imshow(amplitude, cmap="Greys")
    axs[1, 1].set(title="Edge magnitude")
    fig.colorbar(amp_plot, ax=axs[1, 1])

    dir_plot = axs[1, 2].imshow(
        direction,
        cmap="coolwarm",
        interpolation="nearest",
        norm=colors.Normalize(-180.0, 180.0),
    )
    axs[1, 2].set(title="Sobel edge direction")
    dir_cb = fig.colorbar(dir_plot, ax=axs[1, 2])
    dir_cb.set_label("Degrees")

    fig.delaxes(axs[1, 0])

    return fig
