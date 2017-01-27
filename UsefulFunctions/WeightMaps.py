import numpy as np
from skimage.morphology import binary_dilation
from skimage.morphology import disk
import operator
import glob
import os
from scipy import misc
from UsefulFunctions.RandomUtils import CheckOrCreate, CheckExistants, CheckFile
from UsefulFunctions.ImageTransf import Identity
from scipy import ndimage
import WrittingTiff.tifffile as tif

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
    new_res, bin_image = ComputeDistanceImage(lbl, thresh=thresh)
    new_lbl = FrequencyBalanceMatrix(lbl, val)

    new_res = np.exp(- np.square(new_res) / (2 * sigma * sigma))
    new_res[bin_image == 1] = 0

    return new_lbl + w_0 * new_res

# val = FrequencyBalanceValues(datagen)
import skimage.measure as skm


def ComputeWeightMap(input_path, w_0, val, sigma=5):

    from Data.DataGen import DataGen
    WGT_DIR = os.path.join(input_path, "WEIGHTS")
    CheckOrCreate(WGT_DIR)
    wgt_dir = os.path.join(
        WGT_DIR, "{}_{}_{}_{}".format(w_0, val[0], val[1], sigma))
    CheckOrCreate(wgt_dir)
    datagen = DataGen(input_path, transforms=[Identity()])
    datagen.SetPatient("0000000")
    folder = glob.glob(os.path.join(input_path, 'GT_*'))

    for fold in folder:
        num = fold.split('_')[-1].split(".")[0]
        new_fold = os.path.join(wgt_dir, "WGT_" + num)
        CheckOrCreate(new_fold)
        list_nii = glob.glob(os.path.join(fold, '*.nii.gz'))
        for name in list_nii:
            # print name
            lbl = datagen.LoadGT(name)
            lbl_diff = skm.label(lbl)
            res = WeightMap(lbl_diff, w_0, val, sigma)
            name = os.path.join(new_fold, name.split('/')[-1])
            new_name = name.replace('nii.gz', 'png')
            # print new_name
            misc.imsave(new_name, res)
    return wgt_dir

def ComputeWeightMapIsbi(lbl_path, w_0, val, sigma=5):

    input_path = '/' + os.path.join(*lbl_path.split('/')[:-1])
    WGT_DIR = os.path.join(input_path, "WEIGHTS")
    CheckOrCreate(WGT_DIR)
    wgt_file = os.path.join(
    WGT_DIR, "{}_{}_{}_{}.tif".format(w_0, val[0], val[1], sigma))
    lbl = tif.imread(lbl_path)
    wgt = np.zeros_like(lbl)
    for i in range(wgt.shape[0]):
        print "{} / {}".format(i, wgt.shape[0])
        lbl_diff = skm.label(lbl[i,:,:])
        wgt[i,:,:] = WeightMap(lbl_diff, w_0, val, sigma)
    tif.imsave(wgt_file, wgt)
    return wgt_file
