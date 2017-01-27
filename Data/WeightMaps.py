
import numpy as np
from skimage.morphology import binary_dilation
from skimage.morphology import disk
import operator


def FrequencyBalanceValues(datagen):

    key = datagen.RandomKey(False)
    img, lbl = datagen[key]
    key = datagen.NextKey(key)
    lbl[lbl > 0] = 1

    val = np.sum(lbl)
    ones = val
    nber_pix = lbl.shape[0] * lbl.shape[1]
    zeros = nber_pix - val

    for i in range(1, datagen.length):
        img, lbl = datagen[key]
        key = datagen.NextKey(key)

        lbl[lbl > 0] = 1

        val = np.sum(lbl)
        ones += val
        zeros += zeros
    return 1.0, (zeros + 0.0) / ones


def FrequencyBalanceMatrix(lbl, val):
    if len(val) != 2:
        raise Exception("val need two values")
    new_lbl = lbl.copy()
    new_lbl[new_lbl > 0] = val[1]
    new_lbl[new_lbl == 0] = val[0]
    return new_lbl


def ComputeDistanceImage(lbl, thresh=1000, theta=1, dilation=True):
    # bin_image stores the new eroded (on the labels) binary mask (GT) image
    bin_image = np.zeros(shape=(lbl.shape[0], lbl.shape[1]))

    num_pixels = lbl.shape[0] * lbl.shape[1]
    num_col = np.max(lbl) - 1
    pixel_matrix = np.zeros(shape=(num_pixels, num_col))
    pixel_matrix = pixel_matrix.astype(np.float32)
    for num in range(1, num_col):
        new = lbl.copy()
        new[new != num] = -1
        new[new == num] = 0
        new[new == -1] = 1
       # pdb.set_trace()
        if dilation:
            selem = disk(theta)
            new = binary_dilation(new, selem=selem)
            new = new.astype(np.uint8)
            x, y = np.where(new == 0)
            bin_image[x, y] = 1

        distance = ndimage.distance_transform_edt(new)
        distance[distance > thresh] = -1
        vec = distance.flatten('C')
        pixel_matrix[:, num - 1] = vec

    def Compute2max(x):
        if np.max(x) == 0:
            return 0
        else:
            new_x = x[x > 0]
            index_d1, d1 = min(enumerate(new_x), key=operator.itemgetter(1))
            new_x[index_d1] = 300000000
            d2 = np.min(new_x)
            return (d1 + d2)

    new_res_flat = np.apply_along_axis(Compute2max, axis=1, arr=pixel_matrix)
    new_res = np.reshape(new_res_flat, lbl.shape, order='C')

    return new_res, bin_image


def WeightMap(lbl, w_0, val, sigma=5, thresh=1000):
    pdb.set_trace()
    new_res, bin_image = ComputeDistanceImage(lbl, thresh=thresh)
    new_lbl = FrequencyBalanceMatrix(lbl, val)

    new_res = np.exp(- np.square(new_res) / (2 * sigma * sigma))
    new_res[bin_image == 1] = 0

    return new_lbl + w_0 * new_res
