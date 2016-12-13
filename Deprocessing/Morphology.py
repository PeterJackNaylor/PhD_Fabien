# -*- coding: utf-8 -*-

from skimage.morphology import watershed
import numpy as np
from skimage.measure import label
from skimage.morphology import reconstruction, dilation, erosion, disk, diamond, square
from skimage import img_as_ubyte


def PrepareProb(img, convertuint8=True, inverse=True):
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img


def HreconstructionErosion(prob_img, h):

    def making_top_mask(x, lamb=h):
        if 255 >= x + lamb:
            return x + lamb
        else:
            return 255
    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(
        seed, mask, method='erosion').astype(np.dtype('ubyte'))
    return recons


def find_maxima(img, convertuint8=False, inverse=False):
    img = PrepareProb(img, convertuint8=convertuint8, inverse=inverse)
    recons = HreconstructionErosion(img, 1)
    return recons - img


def GetContours(img):
    """
    The image has to be a binary image 
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))


def generate_wsl(ws):
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    return img_as_ubyte(grad)


def DynamicWatershedAlias(p_img, lamb):
    b_img = (p_img > 0.5) + 0
    Probs_inv = PrepareProb(p_img)

    Hrecons = HreconstructionErosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)

    wsl = generate_wsl(ws_labels)
    b_img[wsl > 0] = 0

    return label(b_img)
