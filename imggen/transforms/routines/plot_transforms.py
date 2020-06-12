from typing import List

from matplotlib import figure
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from skimage import io, transform

from imggen.routines import plottools
import definitions


def plot_transform(defect_file_name: str) -> figure.Figure:
    return _do_plot(defect_file_name)[0]


def plot_error(defect_file_name: str) -> figure.Figure:
    return _do_plot(defect_file_name)[1]


def _do_plot(defect_file_name: str) -> List[figure.Figure]:
    defect_pos_px = pd.read_csv(
        definitions.DATA_DIR / "manual_defects" / defect_file_name, index_col=0
    )
    defect_pos_mm = pd.read_csv(
        definitions.DATA_DIR / "groundtruth_mm.csv", index_col=0
    )

    # Remove the missing defects from the mm points
    defect_pos_mm = defect_pos_mm.loc[defect_pos_px.index]

    # See https://github.com/scikit-image/scikit-image/issues/1749 for why the axes are
    # flipped
    defect_pos_px = np.flip(defect_pos_px.values, axis=-1)
    defect_pos_mm = np.flip(defect_pos_mm.values, axis=-1)

    model: transform.AffineTransform = transform.estimate_transform(
        "affine", defect_pos_px, defect_pos_mm
    )

    grey = io.imread(definitions.DATA_DIR / "stepped.tiff")[:, :, 0]
    warped = transform.warp(grey, model.inverse, output_shape=(250, 800), cval=np.nan)

    warp_fig, warp_ax = plottools.create_subplots(0.35)
    warp_ax.set(title="Full part", xlabel="x (mm)", ylabel="y (mm)")
    warp_ax.imshow(warped)

    predicted = model(defect_pos_px)
    errors = predicted - defect_pos_mm
    q = warp_ax.quiver(
        predicted[:, 0], predicted[:, 1], errors[:, 0], errors[:, 1], width=0.003,
    )
    warp_ax.quiverkey(q, 0.8, 0.9, 5, label="Error (5 mm)", labelpos="E")

    error_norm = np.linalg.norm(errors, axis=-1)
    error_stats = stats.describe(error_norm, 0)
    print(error_stats)

    distribution_fig, distribution_ax = plottools.create_subplots(0.4)

    sns.distplot(error_norm, ax=distribution_ax, hist=False, rug=True)
    distribution_ax.set_xlim(0, distribution_ax.get_xlim()[1])
    distribution_ax.set(xlabel="Error (mm)", ylabel="Frequency")

    return [warp_fig, distribution_fig]
