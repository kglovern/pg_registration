import numpy as np
import skimage as sk
import skimage.io as skio
import scipy.misc as skmisc
import skimage.transform as sktx
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import time
import math
import glob
import os
import warnings

# TOFIX: The low res warnings from conversion.  For now just suppress them
warnings.filterwarnings("ignore")


def getImgHeight(img):
    return np.floor(img.shape[0])


def getOperationTime(start, end):
    return end - start


def cropImg(img, amt=0.1):
    height, width = img.shape
    cropHeight = amt * height
    cropWidth = amt * width
    height = height - cropHeight
    width = width - cropWidth
    return img[int(cropHeight / 2):int(height - (cropHeight / 2)), int(cropWidth / 2):int(width - (cropWidth / 2))]


def getChannelsFromOrig(img):
    img = sk.img_as_float(img)
    imgHeight = getImgHeight(img)

    sliceHeight = int(np.floor(imgHeight / 3))

    blueEnd = sliceHeight
    greenEnd = 2 * sliceHeight
    redEnd = 3 * sliceHeight

    blue = img[:blueEnd]
    green = img[blueEnd:greenEnd]
    # Need redEnd to account for rounding issues leading to one channel being larger than the others
    red = img[greenEnd:redEnd]

    return red, green, blue


def openImg(fPath):
    return skio.imread(fPath)


def getWindow(img, xStart, yStart, width):
    return img[yStart:yStart + width, xStart:xStart + width]


def getDownscaledWindow(img, xStart, yStart, width, scale):
    retImg = getWindow(img, xStart, yStart, width)
    retImg = sktx.rescale(retImg, scale, mode="reflect", anti_aliasing=False, multichannel=False)
    return retImg


def shiftImage(img, xOff, yOff):
    trans = sktx.AffineTransform(translation=(xOff, yOff))
    return sktx.warp(img, trans)


def alignChannels(ref, target, xOff=0.4, yOff=0.4, wSize=0.38, numMoves=24, pyrLevel=4):
    bestX = 0
    bestY = 0
    shiftChanged = False
    maxSSIM = 0

    iWidth, iHeight = ref.shape
    xStart = int(xOff * iWidth)
    yStart = int(yOff * iHeight)
    wWidth = int(wSize * iWidth)

    # Initial downscale at 1/8th resolution - TODO: use Pyrlevel to calculate
    curScale = 1.0 / 8.0

    modRef = ref
    iterationCount = 0
    start = time.time()
    while not curScale > 1:
        print(f"Scale: {curScale}")
        shiftChanged = False
        bestX = 0
        bestY = 0
        dTarget = getDownscaledWindow(target, xStart, yStart, wWidth, curScale)
        for yOff in range(int(-0.5 * numMoves), int(0.5 * numMoves)):
            for xOff in range(int(-0.5 * numMoves), int(0.5 * numMoves)):
                dRef = getDownscaledWindow(modRef, xStart + xOff, yStart + yOff, wWidth, curScale)
                curSSIM = ssim(dRef, dTarget)
                if curSSIM > maxSSIM:
                    shiftChanged = True
                    maxSSIM = curSSIM
                    bestX = xOff
                    bestY = yOff
                    # print (f"New Max at {xOff}, {yOff}: {maxSSIM}")

        # Shift original channel - pyramids
        #xShift = bestX * math.pow(2, (pyrLevel - 1))
        #yShift = bestY * math.pow(2, (pyrLevel - 1))
        xShift = bestX
        yShift = bestY
        # print(f"Shift: {xShift}, {yShift}")
        if shiftChanged:
            modRef = shiftImage(modRef, xShift, yShift)
        curScale = 2 * curScale
        pyrLevel -= 1
    end = time.time()
    opTime = getOperationTime(start, end)
    return modRef, opTime


def saveImg(img, fPath):
    directory = "/".join(fPath.split("/")[:-1])
    if not os.path.isdir(directory):
        os.makedirs(directory)

    skio.imsave(fPath, img)


def showImageAsFigure(img):
    plt.figure()
    skio.imshow(img)
    plt.show()
    plt.clf()


def combineChannels(r, g, b):
    return np.dstack((r, g, b))


def to8Bit(img):
    return sk.img_as_ubyte(img)


def colorizeGorskiiImgNaive(fPath):
    filename = getFileName(fPath)
    oDir = f"results/{filename}"

    img = openImg(fPath)
    # Downsample from 16 to 8 bit grayscale
    img = to8Bit(img)
    rc, gc, bc = getChannelsFromOrig(img)
    combined = combineChannels(rc, gc, bc)
    saveImg(combined, f"{oDir}/naive.png")


def getFileName(fPath):
    fileName = fPath.split('/')[-1]
    return fileName.split('.')[0]


def makeOutputDir(file):
    filename = getFileName(file)
    oDir = os.getcwd() + f"/results/{filename}"
    if not os.path.isdir(oDir):
        os.mkdir(oDir)


def colorizeGorskiiImgWirth(fPath):
    filename = getFileName(fPath)
    oDir = f"results/{filename}"

    img = openImg(fPath)
    rc, gc, bc = getChannelsFromOrig(img)

    # Alteration:  Register both to Red for more accuracy instead of G to R and B to G
    gc, timeG = alignChannels(gc, rc)
    bc, timeB = alignChannels(bc, rc)
    # Channel comparisons
    saveImg(rc, f"{oDir}/red.png")
    saveImg(gc, f"{oDir}/green.png")
    saveImg(bc, f"{oDir}/blue.png")

    # Remove 10% around the border for each channel
    rc = cropImg(rc)
    gc = cropImg(gc)
    bc = cropImg(bc)

    combined = combineChannels(rc, gc, bc)
    # output image that hopefully doesn't suck
    saveImg(combined, f"{oDir}/wirth.png")
    return timeG + timeB


def runOnDir(imgDir):

    curDir = os.getcwd()
    resultsDir = f"{curDir}/results"
    if not os.path.isdir(resultsDir):
        os.mkdir(resultsDir)
    files = glob.iglob(f"{imgDir}/**/*", recursive=True)
    for file in files:
        makeOutputDir(file)
        print(f"\nRunning Naive on {file}")
        colorizeGorskiiImgNaive(file)
        print(f"\nRunning Wirth on {file}")
        opTime = colorizeGorskiiImgWirth(file)
        print(f"Runtime: {opTime}")
