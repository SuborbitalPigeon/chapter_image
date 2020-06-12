from matplotlib import figure

from .routines import plot_transforms


def plot_full_part() -> figure.Figure:
    return plot_transforms.plot_transform("threshold.csv")


def plot_error() -> figure.Figure:
    return plot_transforms.plot_error("threshold.csv")
