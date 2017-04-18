import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    # Calculate corner points when applying homography
    corners = []
    corners.append(np.dot(M, np.array([0,0,1])))
    corners.append(np.dot(M, np.array([0,img.shape[1],1])))
    corners.append(np.dot(M, np.array([img.shape[0],0,1])))
    corners.append(np.dot(M, np.array([img.shape[0],img.shape[1],1])))

    # Convert back to cartesian coordinates
    corners = [corner[:2]/corner[2] for corner in corners]

    # Get Xs
    xs = sorted([corner[0] for corner in corners])
    # Get Ys
    ys = sorted([corner[1] for corner in corners])

    # Get the second smallest and largest value for the bounding box
    minX = xs[1]
    minY = ys[1]
    maxX = xs[-2]
    maxY = ys[-2]
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    # feathering
    if (maxX - minX) < 2*blendWidth:
        blendWidth = (maxX - minX) / 2 - 1
    alpha = np.concatenate((np.linspace(0., 1., blendWidth),
                            np.ones(maxX - minX - 2*blendWidth),
                            np.linspace(1., 0., blendWidth)))

    # new matrix with space for alpha channel
    withalpha = np.ones((img.shape[0], img.shape[1], 4))

    # move in image channels
    withalpha[:,:,0] = img[:,:,0]
    withalpha[:,:,1] = img[:,:,1]
    withalpha[:,:,2] = img[:,:,2]

    # pad for linear interpolation
    #withalpha = np.pad(withalpha, ((2,2), (2,2), (0,0)), 'edge')

    # inverse transformation matrix
    M_inv = np.linalg.inv(M)

    # NOTE: Using nearest interpolation. Linear interpolation can be used by using flag 'cv2.INTER_LINEAR'.
    # However this creates hairline borders around the individual images.
    warped = cv2.warpPerspective(withalpha, M_inv, (acc.shape[1],acc.shape[0]), flags=(cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST))

    # for column in space of panorama reserved for this image
    for column in range(minX, maxX):
        warped[:, column, :3] = warped[:, column, :3] * alpha[column - minX] # calculate feathered RGB value

        vals = np.full((warped.shape[0]), alpha[column - minX])
        warped[:, column, 3] = vals # assign correct values to opacity channel

        for row in range(minY, maxY):
            if(np.array_equal(warped[row, column, :3], [0,0,0])): # if the pixel is black
                warped[row, column, 3] = 0.0; # set opacity to 0
            acc[row, column] += warped[row, column] # save RGB value of pixel in accumulator

    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    img = np.zeros((acc.shape[0], acc.shape[1], 3), dtype=np.uint8)
    for row in range(acc.shape[0]):
        for column in range(acc.shape[1]):
            if acc[row, column, 3] > 0:
                img[row, column] = (acc[row, column, 0:3] / acc[row, column, 3]).astype(int)
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        minX_box, minY_box, maxX_box, maxY_box = imageBoundingBox(img, M)
        minX = min(minX, minX_box)
        minY = min(minY, minY_box)
        maxX = max(maxX, maxX_box)
        maxY = max(maxY, maxY_box)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN

    if(is360):
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
