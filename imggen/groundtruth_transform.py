import logging
from typing import List

from matplotlib import figure
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from skimage import io, transform

from imggen.routines import plottools
import definitions


log = logging.getLogger(__name__)


def plot_full_part() -> figure.Figure:
    return _plot_groundtruth_transform()[0]


def plot_error() -> figure.Figure:
    return _plot_groundtruth_transform()[1]


def _plot_groundtruth_transform() -> List[figure.Figure]:
    defect_pos_px = pd.read_csv(definitions.DATA_DIR / "stepped-px.csv", index_col=0)
    defect_pos_mm = pd.read_csv(definitions.DATA_DIR / "groundtruth_mm.csv", index_col=0)

    # Remove the missing defect from the mm points
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
    log.info(f"Minimum error is {error_stats[1][0]:.2f} mm")
    log.info(f"Maximum error is {error_stats[1][1]:.2f} mm")
    log.info(f"Average error is {error_stats[2]:.2f} mm")

    x_scale, y_scale = model.scale
    x_trans, y_trans = model.translation

    log.info(f"Translation is {x_trans:.3g} mm in x, and {y_trans:.3g} mm in y")
    log.info(
        f"Scale is {x_scale:.3g} mm/px in the x direction, and {y_scale:.3g} mm/px in y"
    )
    log.info(f"Shear is {model.shear:.3g}")
    log.info(f"Rotation is {model.rotation:.3g} rad")

    distribution_fig, distribution_ax = plottools.create_subplots(0.4)

    sns.distplot(error_norm, ax=distribution_ax, hist=False, rug=True)
    distribution_ax.set_xlim(0, distribution_ax.get_xlim()[1])
    distribution_ax.set(xlabel="Error (mm)", ylabel="Frequency")

    return [warp_fig, distribution_fig]
