# -*- coding: utf-8 -*-

import numpy as np
from skimage.measure import label
from skimage.morphology import reconstruction, dilation, erosion, disk


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
    recons = reconstruction(seed, mask, method='erosion')
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


def DynamicWatershedAlias(p_img, lamb):
    b_img = (p_img > 0.5) + 0
    Probs_inv = PrepareProb(p_img)
    Hrecons = HreconstructionErosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    new_ws_labels = ws_labels.copy()
    for val_m in range(1, np.max(ws_labels) + 1):
        temp = ws_labels.copy()
        temp[temp != val_m] = 0
        temp[temp == val_m] = 1
        Contours = GetContours(temp)
        new_ws_labels[Contours == 1] = 0
    return label(new_ws_labels)
