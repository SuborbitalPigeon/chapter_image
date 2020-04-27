import pathlib
import subprocess

from imggen import equalisation


src_dir = pathlib.Path("text")
img_out_dir = src_dir / "images"
out_dir = src_dir / "out"

# Generate equalise image
fig = equalisation.plot_equalise()
fig.savefig(img_out_dir / "equalise.png")

# Generate text
src_file = src_dir / "image.md"

print("Creating Word document")
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

print("Creating TeX source")
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
