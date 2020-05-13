import inspect
import logging
import pathlib
import subprocess
from typing import Callable

from matplotlib import figure

from imggen import (
    affine,
    equalisation,
    equalise_hist,
    groundtruth_transform,
    threshold,
    sobel,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

src_dir = pathlib.Path("text")
img_out_dir = src_dir / "images"
out_dir = src_dir / "out"


def generate_image(func: Callable[[], figure.Figure], file_name: str) -> None:
    image_path = img_out_dir / file_name

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
    fig.savefig(image_path)


# Generate equalise image
generate_image(affine.plot_affine, "affine.png")
generate_image(equalisation.plot_equalise, "equalise.png")
generate_image(equalise_hist.plot_equalise_hist, "equalise_hist.png")
generate_image(threshold.plot_global_threshold, "global_threshold.png")
generate_image(groundtruth_transform.plot_error, "groundtruth_error.png")
generate_image(groundtruth_transform.plot_full_part, "groundtruth_transform.png")
generate_image(threshold.plot_local_threshold, "local_threshold.png")
generate_image(sobel.plot_sobel, "sobel.png")

# Generate text
src_file = src_dir / "image.md"

log.info("Creating Word document")
subprocess.run(
    [
        "pandoc",
        "-F",
        "pandoc-crossref",
        "-F",
        "pandoc-citeproc",
        "--reference-doc",
        str(src_dir / "num-reference.docx"),
        f"--resource-path={src_dir}",
        str(src_file),
        "-o",
        (str(out_dir / "Image.docx")),
    ]
)

log.info("Creating TeX source")
subprocess.run(
    [
        "pandoc",
        "-F",
        "pandoc-crossref",
        "--biblatex",
        f"--resource-path={src_dir}",
        src_file,
        "-N",
        "-o",
        (str(out_dir / "image.tex")),
    ]
)
