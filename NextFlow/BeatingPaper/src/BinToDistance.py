import nibabel as ni
import pdb
from skimage.io import imread, imsave
from glob import glob
from os.path import dirname, join, basename
from shutil import copy
from UsefulFunctions.RandomUtils import CheckOrCreate
from scipy.ndimage.morphology import distance_transform_cdt
from skimage.measure import label, regionprops
from Deprocessing.Morphology import generate_wsl
import pdb
import numpy as np
from skimage.io import imread

def LoadGT(path):
    img = imread(path, dtype='uint8')
    return img

def DistanceWithoutNormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res

NEW_FOLDER = 'ToAnnotateDistance'
CheckOrCreate(NEW_FOLDER)
for image in glob('ForDataGenTrainTestVal/Slide_*/*.png'):
    baseN = basename(image)
    Slide_name = dirname(image)
    GT_name = baseN.replace('Slide', 'GT')
    OLD_FOLDER = dirname(Slide_name)
    Slide_N = basename(dirname(image))
    GT_N = Slide_N.replace('Slide_', 'GT_')
    
    CheckOrCreate(join(NEW_FOLDER, Slide_N))
    CheckOrCreate(join(NEW_FOLDER, GT_N))

    copy(image, join(NEW_FOLDER, Slide_N, baseN))
    bin_image = LoadGT(join(OLD_FOLDER, GT_N, GT_name))
    res = DistanceWithoutNormalise(bin_image)
    imsave(join(NEW_FOLDER, GT_N, GT_name), res)
