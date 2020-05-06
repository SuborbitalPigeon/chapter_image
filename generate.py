import logging
import pathlib
import subprocess

from imggen import affine, equalisation, equalise_hist, threshold, sobel

log = logging.getLogger(__name__)
src_dir = pathlib.Path("text")
img_out_dir = src_dir / "images"
out_dir = src_dir / "out"

logging.basicConfig(level=logging.INFO)

# Generate equalise image
log.info("Generating equalisation image")
fig = equalisation.plot_equalise()
fig.savefig(img_out_dir / "equalise.png")

log.info("Generating equalisation histogram image")
fig = equalise_hist.plot_equalise_hist()
fig.savefig(img_out_dir / "equalise_hist.png")

log.info("Generating sobel image")
fig = sobel.plot_sobel()
fig.savefig(img_out_dir / "sobel.png")

log.info("Generating affine transform image")
fig = affine.plot_affine()
fig.savefig(img_out_dir / "affine.png")

log.info("Generating global thresholding image")
fig = threshold.plot_global_threshold()
fig.savefig(img_out_dir / "global_threshold.png")

log.info("Generating local thresholding image")
fig = threshold.plot_local_threshold()
fig.savefig(img_out_dir / "local_threshold.png")

# Generate text
src_file = src_dir / "image.md"

log.info("Creating Word document")
subprocess.run(
    [
        "pandoc",
        "-F", "pandoc-crossref",
        "-F", "pandoc-citeproc",
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
