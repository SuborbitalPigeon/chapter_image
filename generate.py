import inspect
import logging
import pathlib
import subprocess
from typing import Callable

from matplotlib import figure

from imggen import (
    affine,
    all_threshold,
    decompose,
    denoise,
    equalisation,
    equalise_hist,
    groundtruth_transform,
    threshold,
    sobel,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SRC_DIR = pathlib.Path("text")
IMG_OUT_DIR = SRC_DIR / "images"
OUT_DIR = SRC_DIR / "out"


def generate_image(func: Callable[[], figure.Figure], file_name: str) -> None:
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
    fig = func()
    fig.savefig(image_path, transparent=True)


# Generate equalise image
generate_image(threshold.plot_adaptive_threshold, "adaptive_threshold.svg")
generate_image(affine.plot_affine, "affine.svg")
generate_image(all_threshold.plot_all_thresholds, "all_threshold.svg")
generate_image(decompose.plot_wavelet_decompose, "decompose.svg")
generate_image(denoise.plot_denoise_all, "denoise_all.svg")
generate_image(denoise.plot_denoise_gauss, "denoise_gauss.svg")
generate_image(denoise.plot_denoise_median, "denoise_median.svg")
generate_image(denoise.plot_denoise_nlmeans, "denoise_nlmeans.svg")
generate_image(denoise.plot_denoise_wavelet, "denoise_wavelet.svg")
generate_image(equalisation.plot_equalise, "equalise.svg")
generate_image(equalise_hist.plot_equalise_hist, "equalise_hist.svg")
generate_image(threshold.plot_global_threshold, "global_threshold.svg")
generate_image(groundtruth_transform.plot_error, "groundtruth_error.svg")
generate_image(groundtruth_transform.plot_full_part, "groundtruth_transform.svg")
generate_image(sobel.plot_sobel, "sobel.svg")

# Generate text
src_file = SRC_DIR / "image.md"

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

log.info("Creating OpenDocument")
subprocess.run(
    [
        "pandoc",
        "-F",
        "pandoc-crossref",
        "-F",
        "pandoc-citeproc",
        "--reference-doc",
        str(SRC_DIR / "num-reference.odt"),
        f"--resource-path={SRC_DIR}",
        str(src_file),
        "-o",
        (str(OUT_DIR / "Image.odt")),
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
