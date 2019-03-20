import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktx
import matplotlib.pyplot as plt

def getImgHeight(img):
    return np.floor(img.shape[0])

def getChannelsFromOrig(img):
    img = sk.img_as_float(img)
    imgHeight = getImgHeight(img)
    
    sliceHeight = int(np.floor(imgHeight / 3))

    blueEnd = sliceHeight
    greenEnd = 2 * sliceHeight
    redEnd = 3 * sliceHeight

    blue = img[:blueEnd]
    green = img[blueEnd:greenEnd]
    red = img[greenEnd:redEnd]

    return red, green, blue

def openImg(fPath):
    return skio.imread(fPath)

def colorizeGorskiiImg(fPath):
    img = openImg(fPath)
    rc, gc, bc = getChannelsFromOrig(img)
    plt.figure()
    combined = np.dstack((rc, gc, bc))
    skio.imshow(combined)
    plt.show()
    plt.clf()



