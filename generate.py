import inspect
import logging
import pathlib
import subprocess
from typing import Any, Callable, Optional

from matplotlib import figure, pyplot as plt

from imggen import (
    affine,
    all_threshold,
    decompose,
    denoise,
    equalisation,
    equalise_hist,
    sobel,
    threshold,
)
from imggen import pipelines, transforms

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SRC_DIR = pathlib.Path("text")
IMG_OUT_DIR = SRC_DIR / "images"
OUT_DIR = SRC_DIR / "out"


def generate_image(
    func: Callable[[Optional[Any]], figure.Figure], file_name: str, **kwargs: Any
) -> None:
    image_path = IMG_OUT_DIR / file_name

    try:
        image_mtimens = image_path.stat().st_mtime_ns
        func_mtimens = pathlib.Path(inspect.getfile(func)).stat().st_mtime_ns
    except FileNotFoundError:
        pass
    else:
        if image_mtimens > func_mtimens:
            log.info(f"Skipping generating {file_name} as code hasn't changed")
            return

    log.info(f"Generating {file_name}")
    fig = func(**kwargs)
    fig.savefig(image_path, transparent=True)
    plt.close(fig)


# Generate images
image_type = "png"

generate_image(threshold.plot_adaptive_threshold, "adaptive_threshold." + image_type)
generate_image(affine.plot_affine, "affine." + image_type)
generate_image(all_threshold.plot_all_thresholds, "all_threshold." + image_type)
generate_image(decompose.plot_wavelet_decompose, "decompose." + image_type)
generate_image(denoise.plot_denoise_all, "denoise_all." + image_type)
generate_image(equalisation.plot_equalise, "equalise." + image_type)
generate_image(equalise_hist.plot_equalise_hist, "equalise_hist." + image_type)
generate_image(threshold.plot_global_threshold, "global_threshold." + image_type)
generate_image(sobel.plot_sobel, "sobel." + image_type)

# Pipeline images
pipelines_dir = pathlib.Path("pipelines")
generate_image(pipelines.canny.plot_threshold, pipelines_dir / ("canny." + image_type))
generate_image(
    pipelines.canny.plot_extra_threshold, pipelines_dir / ("canny_extra." + image_type)
)
generate_image(
    pipelines.threshold.plot_threshold, pipelines_dir / ("threshold." + image_type)
)
generate_image(
    pipelines.threshold.plot_extra_threshold,
    pipelines_dir / ("threshold_triple." + image_type),
)

# Transform images
transforms_dir = pathlib.Path("transforms")

generate_image(
    transforms.canny.plot_error, transforms_dir / ("canny_error." + image_type)
)
generate_image(
    transforms.canny.plot_full_part, transforms_dir / ("canny." + image_type)
)

generate_image(
    transforms.fill.plot_error, transforms_dir / ("fill_error." + image_type)
)
generate_image(transforms.fill.plot_full_part, transforms_dir / ("fill." + image_type))

generate_image(
    transforms.groundtruth.plot_error,
    transforms_dir / ("groundtruth_error." + image_type),
)
generate_image(
    transforms.groundtruth.plot_full_part,
    transforms_dir / ("groundtruth." + image_type),
)

generate_image(
    transforms.threshold.plot_error, transforms_dir / ("threshold_error." + image_type)
)
generate_image(
    transforms.threshold.plot_full_part, transforms_dir / ("threshold." + image_type)
)

# Generate text
src_file = SRC_DIR / "image.md"

# SVG doesn't work: https://github.com/jgm/pandoc/issues/4058
log.info("Creating Word document")
subprocess.run(
    [
        "pandoc",
        "-F",
        "pandoc-crossref",
        "-F",
        "pandoc-citeproc",
        "--reference-doc",
        str(SRC_DIR / "num-reference.docx"),
        f"--resource-path={SRC_DIR}",
        str(src_file),
        "-o",
        (str(OUT_DIR / "Image.docx")),
    ]
)

log.info("Creating TeX source")
subprocess.run(
    [
        "pandoc",
        "-F",
        "pandoc-crossref",
        "--biblatex",
        f"--resource-path={SRC_DIR}",
        src_file,
        "-N",
        "-o",
        (str(OUT_DIR / "image.tex")),
    ]
)
