import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktx
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
    start = time.time()
    img = sk.img_as_float(img)
    imgHeight = getImgHeight(img)
    
    sliceHeight = int(np.floor(imgHeight / 3))

    blueEnd = sliceHeight
    greenEnd = 2 * sliceHeight
    redEnd = 3 * sliceHeight

    blue = img[:blueEnd]
    green = img[blueEnd:greenEnd]
    red = img[greenEnd:redEnd]
    
    end = time.time()
    printOperationTime(start, end)

    return red, green, blue

def openImg(fPath):
    return skio.imread(fPath)

def getWindow(img, xStart, yStart, width):
    return img[yStart:yStart+width, xStart:xStart+width]

def getDownscaledWindow(img, xStart, yStart, width, scale):
    retImg = getWindow(img, xStart, yStart, width)
    retImg = sktx.rescale(retImg, scale, mode="reflect", anti_aliasing=False, multichannel=False) 
    return retImg


def alignChannels(ref, target, xOff=0.4, yOff=0.4, wSize=0.38, numMoves=24):
    bestX = 0
    bestY = 0
    maxSSIM = 0
    pyrLevel = 4
    
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
        dTarget = getDownscaledWindow(target, xStart, yStart, wWidth, curScale) 
        for xOff in range(int(-0.5 * numMoves), int(0.5 * numMoves)):
            for yOff in range (int(-0.5 * numMoves), int(0.5 * numMoves)): 
                iterationCount += 1
                dRef = getDownscaledWindow(modRef, xStart + xOff, yStart + yOff, wWidth, curScale)
                curSSIM = ssim(dRef, dTarget)
                if curSSIM > maxSSIM:
                    maxSSIM = curSSIM
                    bestX = xOff
                    bestY = yOff
                    print (f"New Max at {xOff}, {yOff}: {maxSSIM}")

        # Shift original channel
        xShift = bestX * math.pow(2, (pyrLevel - 1))
        yShift = bestY * math.pow(2, (pyrLevel - 1))
        print(f"Shifts: {xShift}, {yShift}")
        # DO THE SHIFT
        curScale = 2 * curScale
    print (f"Iterations: {iterationCount}")
    end = time.time()
    printOperationTime(start, end)
    return ref

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

def colorizeGorskiiImgNaive(fPath):
    img = openImg(fPath)
    rc, gc, bc = getChannelsFromOrig(img)
    combined = combineChannels(rc, gc, bc)
    showImageAsFigure(combined)
    #saveImg("output.png", combined)

def colorizeGorskiiImgWirth(fPath):
    img = openImg(fPath)
    rc, gc, bc = getChannelsFromOrig(img)
    bc = alignChannels(bc, gc)
    #gc = alignChannels(gc, rc)
    combined = combineChannels(rc, gc, bc)
    # TODO: Crop
    #showImageAsFigure(combined)



