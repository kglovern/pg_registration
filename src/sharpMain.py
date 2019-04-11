"""
Getting the rank of the image will give us a noise metric.

Assumes that you are running from the root of the project.
"""
import math
from pathlib import Path
import time

import numpy as np
import skimage.io as skio

import blurMetric

# Modify this to only do the files you want
IMAGES_TO_RANK = [
    *list(Path("results/").rglob("wirth.png")),
    *list(Path("images/Blaise").rglob("*.tif"))
]

def main():
    # Walk through all of the files that we want
    for p in IMAGES_TO_RANK:
        # Get the starting time
        startTime = time.time()

        # Read in the image
        img = skio.imread(p, as_gray=True)
        # Get the blur metric for the luminance
        blurMetricResult = blurMetric.perblurMetric(img)

        # Get the time spent doing the ranking
        duration = time.time() - startTime

        # Print out the results
        print(f"""Got the metrics for {p}
    Perblur Metric: {blurMetricResult:.4f}
    Time: {duration:.4f}""")

if __name__ == "__main__":
    main()
