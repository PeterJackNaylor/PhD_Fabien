# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""
import pdb

from optparse import OptionParser

from skimage.morphology import disk, opening, closing
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label

from scipy.ndimage.morphology import binary_fill_holes, morphological_gradient
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

import UsefulOpenSlide as UOS
plt.rcParams['figure.figsize'] = 26, 26


def Opening(image, iteration=7):
    # computes several openings with a bigger disk at each time..
    inter_image = image.copy()
    inter_image = opening(inter_image, disk(iteration))
    return inter_image


def TissueThresh(image, thresh=None):
    # tissue is in black... Tries to find the best background
    if thresh is None:
        thresh = threshold_otsu(image)
    binary = image > thresh
    if binary.dtype == 'bool':
        binary = binary + 0
    return binary


def FillHole(bin_image, invert=False):
    # files holes
    values = np.unique(bin_image)
    if len(values) > 2:
        print "Not binary image"
        return []
    background = min(values)
    bin_image -= background
    bin_image[bin_image > 0] = 1
    if invert:
        bin_image -= 1
        bin_image[bin_image < 0] = 1
    result = np.copy(bin_image)
    binary_fill_holes(bin_image, output=result)
    return result


def RemoveBorder(image_bin, border=15):
    # removes borders
    neg_border = border * -1
    result = np.copy(image_bin)
    result[:, :] = 0
    result[border:neg_border, border:neg_border] = image_bin[
        border:neg_border, border:neg_border]
    return result


def RemoveIsolatedPoints(binary_image, thresh=100):
    # removing tiny areas...
    # pdb.set_trace()
    labeled = label(binary_image)
    reg_prop = regionprops(labeled)
    for i in range(len(reg_prop)):
        if reg_prop[i].area < thresh:
            labeled[labeled == (i + 1)] = 0
    labeled[labeled > 0] = 1
    return labeled


def FindTicket(RGB_image, _3tuple=(80, 80, 80)):
    # Find the "black ticket on the images"
    temp_image_3 = np.copy(RGB_image)
    temp_image_3[:, :, :] = 0
    for i in range(3):
        temp_image_1 = np.zeros(shape=RGB_image.shape[0:2])
        temp_image_1[np.where(RGB_image[:, :, i] < _3tuple[i])] = 1
        temp_image_3[:, :, i] = temp_image_1

    temp_resultat = temp_image_3.sum(axis=2)
    temp_resultat[temp_resultat > 0] = 1
    #temp_resultat = Filling_holes_2(temp_resultat)
    temp_resultat = closing(temp_resultat, disk(20))
    temp_resultat = opening(temp_resultat, disk(20))
    temp_resultat = RemoveBorder(temp_resultat)
    return temp_resultat


def Preprocessing(image, thresh=200, invert=True):
    return RemoveIsolatedPoints(RemoveBorder(FillHole(TissueThresh(Opening(image), thresh), invert=invert)))


def Contours(bin_image, contour_size=3):
    # Computes the contours
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad


def combining(numpy_array):
    res = np.sum(numpy_array, axis=2)
    res[res > 0] = 1
    return res


def save(original, contour=None, name="random_picture.png"):

    plt.imshow(original)
    if contour is not None:
        x, y = np.where(contour > 0)
        plt.plot(y, x, '.r')
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def ROI_binary_mask(sample, size=5):
    #  very slow function at resolution 4
    PreprocRes = np.copy(sample)

    for i in range(3):  # RGB
        # this one is painfully slow..
        PreprocRes[:, :, i] = Preprocessing(sample[:, :, i])
    res = combining(PreprocRes)
    ticket = FindTicket(sample)
    res = res - ticket
    res[res > 0] = 1
    res[res < 0] = 0
    res = opening(res, disk(size))
    return res

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-d", "--data", dest="data_folder",
                      help="Input folder (raw data)")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Output folder (Images and ROI)")
    parser.add_option('-r', '--res', dest="resolution",
                      help="Resolution")

    (options, args) = parser.parse_args()

    if options.data_folder is None:
        options.data_folder = "D:/dataThomas/Projet_FR-TNBC-2015-09-30/All/*tiff"
    elif 'tiff' not in options.data_folder:
        options.data_folder = os.path.join(options.data_folder, "*.tiff")

    if options.output_folder is None:
        options.output_folder = "D:/dataThomas/Projet_FR-TNBC-2015-09-30/ROI"

    if options.resolution is None:
        options.resolution = 7
    else:
        options.resolution = int(options.resolution)

    print "Input paramters to TissueSegmentation:"
    print " \n "
    print "Input folder     : | " + options.data_folder
    print "Output folder    : | " + options.output_folder
    print "Resolution       : | " + str(options.resolution)

    # checking output folder
    try:
        if not os.path.isdir(options.output_folder):
            os.mkdir(options.output_folder)
    except:
        print "Failed to check output folder..."
    # checking data folder
    try:
        data_files = glob.glob(options.data_folder)
        print "Number of images : | " + str(len(data_files))
        if not len(data_files) > 1:
            print "Maybe a problem, but their is less than one file to analyse..."
    except:
        print "Failed to get input data...."

    print ' \n '
    print "Beginning analyse:"

    start_time = time.time()
# Core of the code
########################################

    for file_tiff in data_files:
        # pdb.set_trace()
        sample = np.array(UOS.GetWholeImage(
            file_tiff, level=options.resolution))[:, :, 0:3]
        res = ROI_binary_mask(sample)
        cont = Contours(res)

        name_tag = file_tiff.split('\\')[-1].split('.')[0]
        file_name = os.path.join(options.output_folder, name_tag + ".png")
        save(sample, cont, file_name)
        save(res, name=os.path.join(options.output_folder, name_tag + "_binary.png"))
########################################

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for all:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image:"
    diff_time = diff_time / len(data_files)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
