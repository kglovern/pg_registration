"""
This file crops the Blaise images so that they will be more similar to ours

Don't run this program more than once as it will crop the cropped pictures
"""
from pathlib import Path

import numpy as np

import BunchLib

def cropBlaise(img):
    # Crop each channel separately
    croppedRed = BunchLib.cropImg(img[:, :, 0])
    croppedGreen = BunchLib.cropImg(img[:, :, 1])
    croppedBlue = BunchLib.cropImg(img[:, :, 2])

    # Put them on top of each other
    return np.array([croppedRed, croppedGreen, croppedBlue])

def main():
    # Find the Blaise images
    blaisePaths = list(Path("images/Blaise").glob("*.tif"))
    # Read in all of the images
    images = map(BunchLib.openImg, blaisePaths)
    # Crop the images
    croppedImages = map(cropBlaise, images)
    # Save the images
    newNames = map(lambda path: str(path.with_name(f"Cropped-{path.name}")), blaisePaths)
    # Invoke the mapping
    list(map(BunchLib.saveImg, croppedImages, newNames))

if __name__ == "__main__":
    main()
