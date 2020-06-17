import logging

import cueimgproc

import definitions

logging.basicConfig(level=logging.INFO)


def main():
    img = cueimgproc.Image.open(definitions.DATA_DIR / "stepped.tiff")
    binary = img.filter(
        cueimgproc.CannyFilter(),
        cueimgproc.BinaryFilter(),
        cueimgproc.RemoveSmallObjectsFilter(32),
    )

    labelled_image = cueimgproc.LabelledImage(binary.data, img)

    regions = labelled_image.region_properties("centroid", "eccentricity")
    regions = regions[regions["eccentricity"] < 0.9]

    regions.to_csv(definitions.PROJECT_ROOT / "datagen" / "raw_data" / "canny.csv")


if __name__ == "__main__":
    main()
