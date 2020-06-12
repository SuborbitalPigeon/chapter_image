import logging

import cueimgproc
import numpy as np

import definitions

logging.basicConfig(level=logging.INFO)


def main():
    img = cueimgproc.Image.open(definitions.DATA_DIR / "stepped.tiff")
    img = img.filter(cueimgproc.RemoveAlphaFilter())
    block_stats = cueimgproc.BlockStatistics(8)
    block_stats(img)

    mean_block_stdev = np.mean(block_stats.stdev)[0]
    trans = cueimgproc.RegionGrowing(mean_block_stdev)
    trans(img)

    props = trans.regions.region_properties("centroid", "eccentricity", "area")
    props = props[props["eccentricity"] < 0.9]

    props.to_csv(definitions.PROJECT_ROOT / "datagen" / "raw_data" / "fill.csv")


if __name__ == "__main__":
    main()
