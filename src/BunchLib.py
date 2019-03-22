import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktx
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import time
import math

def getImgHeight(img):
    return np.floor(img.shape[0])

def printOperationTime(start, end):
    time =  end - start
    print(f"Operation took {time} seconds")

def cropImg(img, amt=0.1):
    pass

def getChannelsFromOrig(img):
    img = sk.img_as_float(img)
    imgHeight = getImgHeight(img)

    sliceHeight = int(np.floor(imgHeight / 3))

    blueEnd = sliceHeight
    greenEnd = 2 * sliceHeight
    redEnd = 3 * sliceHeight

    blue = img[:blueEnd]
    green = img[blueEnd:greenEnd]
    red = img[greenEnd:redEnd] # Need redEnd to account for rounding issues leading to one channel being larger than the others


    return red, green, blue

def openImg(fPath):
    return skio.imread(fPath)

def getWindow(img, xStart, yStart, width):
    return img[yStart:yStart+width, xStart:xStart+width]

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
    shiftChanged = 0
    maxSSIM = 0

    iWidth, iHeight = ref.shape
    xStart = int(xOff * iWidth)
    yStart = int(yOff * iHeight)
    wWidth = int(wSize * iWidth)

    # Initial downscale at 1/8th resolution
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
            for xOff in range (int(-0.5 * numMoves), int(0.5 * numMoves)):
                dRef = getDownscaledWindow(modRef, xStart + xOff, yStart + yOff, wWidth, curScale)
                curSSIM = ssim(dRef, dTarget)
                if curSSIM > maxSSIM:
                    shiftChanged = True
                    maxSSIM = curSSIM
                    bestX = xOff
                    bestY = yOff
                    print (f"New Max at {xOff}, {yOff}: {maxSSIM}")

        # Shift original channel - pyramids
        #xShift = bestX * math.pow(2, (pyrLevel - 1))
        #yShift = bestY * math.pow(2, (pyrLevel - 1))
        xShift = bestX
        yShift = bestY
        print(f"Shift: {xShift}, {yShift}")
        if shiftChanged:
            modRef = shiftImage(modRef, xShift, yShift)
        curScale = 2 * curScale
        pyrLevel -= 1
    end = time.time()
    printOperationTime(start, end)
    return modRef

def saveImg(img, fPath):
    skio.imsave(fPath, img)

def cropImgFromCenter(img, percentage=0.1):
    pass

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
    img = openImg(fPath)
    # Downsample from 16 to 8 bit grayscale
    img = to8Bit(img)
    rc, gc, bc = getChannelsFromOrig(img)
    combined = combineChannels(rc, gc, bc)
    saveImg(combined, "output_naive.png")

def colorizeGorskiiImgWirth(fPath):
    img = openImg(fPath)
    rc, gc, bc = getChannelsFromOrig(img)
    bc = alignChannels(bc, gc)
    gc = alignChannels(gc, rc)
    # Channel comparisons
    saveImg (rc, "red.png")
    saveImg(gc, "green.png")
    saveImg(bc, "blue.png")
    combined = combineChannels(rc, gc, bc)
    # output image that hopefuilly doesn't suck
    saveImg(combined, "output_wirth.png")



