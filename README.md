# Readme

Note that we did not include the high-definition versions of the images in this submission.  This is due to file size concerns.  They are available online if you want to re-run that particular set yourself.

# Dependencies

skimage
numpy
scipy (should be included with skimage)

This library assumes you're using Python 3, specifically a version 3.6 or greater

# To run

In order to run a specific image through the algorithm, you can issues the following command

python3 src/main.py <path/to/image.jpg>

The resulting images will be places in results/<filename>/

A number of images are generated - each of the colour channels are exported separately, along with two composites.  naive.png is a naive version where the channels are just overlapped.  wirth.png is the result of the registration.

If you want to run a set of images, just point the script at an entire directory.

"python3 src/main.py <path/to/images/>


# Source

main.py is a wrapper that calls our library.

All functionality is located in BunchLib.py.  Relevant functions that you might actually care about include:

- alignChannels - our actual alignment algorithm
- colorizeGorskiiImgWirth - wrapper that calls the above and exports channels
- colorizeGorskiiImgNaive - wrapper that creates the naive version of the image

The rest are largely helper functions and import/export functionality