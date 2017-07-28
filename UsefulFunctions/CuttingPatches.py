# -*- coding: utf-8 -*-
"""
Command line example :
%run CuttingPatches.py --file D:/dataThomas/Projet_FR-TNBC-2015-09-30/All\1572_HES_(Colo)_20150925_162438.tiff --output_folder D:/dataThomas/Projet_FR-TNBC-2015-09-30/SlideSegmentation --res 0 --height 512 --width 512 --nber_squares 400 --perc_contour 0.15
%python CuttingPatches.py --file /media/naylor/Peter-HD/dataThomas/Projet_FR-TNBC-2015-09-30/All/Biopsy/498959.tiff --output_folder /media/naylor/Peter-HD/dataThomas/Projet_FR-TNBC-2015-09-30/SlideSegmentation/ --res 0 --height 512 --width 512 --nber_squares 400 --perc_contour 0.15

"""

import pdb

import time
import os
import random
import openslide
import numpy as np
from scipy import ndimage
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import UsefulOpenSlide as UOS
from TissueSegmentation import ROI_binary_mask, save
from Deprocessing.Morphology import GetContours
from optparse import OptionParser
from skimage import measure
import itertools
from ImageTransf import flip_horizontal
import progressbar

def Sample_imagette(im_bin, N, slide, level_resolution, nber_pixels, current_level, mask):
    # I should modify this function, so that the squares don't fall on each
    # other..
    y, x = np.where(im_bin > 0)
    n = len(x)
    indices = range(n)
    random.shuffle(indices)
    # indices=indices[0:N]
    result = []
    i = 0
    # pdb.set_trace()
    while i < n and len(result) < N:
        x_i = x[indices[i]]
        y_i = y[indices[i]]
        if mask[y_i, x_i] == 0:
            para = UOS.find_square(
                slide, x_i, y_i, level_resolution, nber_pixels, current_level)
            result.append(para)
            x_, y_ = UOS.get_X_Y_from_0(slide, para[0], para[1], current_level)
            w_, h_ = UOS.get_size(slide, para[2], para[
                                  3], level_resolution, current_level)
            add = int(w_ / 2)  # allowing how much overlapping?
            mask[max((y_ - add), 0):min((y_ + add + h_), im_bin.shape[0]),
                 max((x_ - add), 0):min((x_ + add + w_), im_bin.shape[1])] = 1
        i += 1
    return(result, mask)


def White_score(slide, para, options):
    crop = UOS.GetImage(slide, para)
    # pdb.set_trace()
    thresh = options["Threshold"]
    crop = np.array(crop)[:, :, 0]
    binary = crop > thresh
    nber_ones = sum(sum(binary))
    nber_total = binary.shape[0] * binary.shape[1]
    return(float(nber_ones) / nber_total)

def CalculateMarge(marge, size_x, size_y):
    return marge, marge

def Best_Finder_rec(slide, level, x_0, y_0, size_x, size_y, ref_level, list_roi, number_of_pixels_max, marge, options):
    #pdb.set_trace()
    if size_x * size_y == 0:
        print 'Warning: width or height is null..'
        return []

    if level == ref_level:
        if size_x * size_y < number_of_pixels_max:

            ## if ref_level != 0 then thir will need some changes..
            width_xp, height_xp = CalculateMarge(marge, size_x, size_y)
            max_width, max_height = slide.dimensions


            x_0 = max(x_0 - width_xp, 0)
            y_0 = max(y_0 - height_xp, 0)
            size_x = min(x_0 + size_x + 2 * width_xp, max_width) - x_0
            size_y = min(y_0 + size_y + 2 * height_xp, max_height) - y_0

            para = [x_0, y_0, size_x, size_y, level]
            val = White_score(slide, para, options)
            if val < options["value"]:
                list_roi.append(para)
            else:
                print val
                print "rejected"
            ## else, you don't append it and it has been discarded
            return list_roi
        else:
            ## image is still too big at the level we intend it to be, so we split it still.
            size_x_new = int(size_x * 0.5)
            size_y_new = int(size_y * 0.5)
            width_x_0, height_y_0 = UOS.get_size(slide, size_x, size_y, level, 0)
            x_1 = x_0 + int(width_x_0 * 0.5)
            y_1 = y_0 + int(height_y_0 * 0.5)
            list_roi = Best_Finder_rec(slide, level, x_0, y_0, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            list_roi = Best_Finder_rec(slide, level, x_1, y_0, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            list_roi = Best_Finder_rec(slide, level, x_0, y_1, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            list_roi = Best_Finder_rec(slide, level, x_1, y_1, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            return list_roi
    elif level > ref_level:
        if size_x * size_y > number_of_pixels_max:
            ### if the image is too big at this level then we can already divide it
            size_x_new, size_y_new = UOS.get_size(slide, size_x, size_y, level, level - 1)
            size_x_new = int(size_x_new * 0.5)
            size_y_new = int(size_y_new * 0.5)
            width_x_0, height_y_0 = UOS.get_size(slide, size_x, size_y, level, 0)
            x_1 = x_0 + int(width_x_0 * 0.5)
            y_1 = y_0 + int(height_y_0 * 0.5)
            list_roi = Best_Finder_rec(slide, level - 1, x_0, y_0, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            list_roi = Best_Finder_rec(slide, level - 1 , x_1, y_0, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            list_roi = Best_Finder_rec(slide, level - 1, x_0, y_1, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            list_roi = Best_Finder_rec(slide, level - 1, x_1, y_1, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            return list_roi

        else:
            #### image is still very small, so we can increase the resolution without any problem
            size_x_new, size_y_new = UOS.get_size(slide, size_x, size_y, level, level - 1)
            list_roi = Best_Finder_rec(slide, level - 1, x_0, y_0, size_x_new, size_y_new,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)
            return list_roi
    else:
        print "Something fucked up"



def ReturnGrid(x_g, x_d, y_h, y_b, fixed_size):
    new_res = []
    list1 = range(x_g, x_d + fixed_size[0], fixed_size[0])
    list2 = range(y_h, y_b + fixed_size[1], fixed_size[1])
    list_tog = list(itertools.product(list1, list2))
    for sub_list in list_tog:
        new_res.append([sub_list[0], sub_list[1],
                        fixed_size[0], fixed_size[1], 0])
    # pdb.set_trace()
    return new_res


def GimmeGrid(slide, res, level=4, fixed_size=(512, 512)):
    lab_res = measure.label(res)
    list_roi = []
    for i in range(1, np.max(lab_res) + 1):
        y_i, x_i = np.where(lab_res == i)
        x_g = np.min(x_i)
        x_d = np.max(x_i)
        y_h = np.min(y_i)
        y_b = np.max(y_i)
       # pdb.set_trace()
        # passing to the lower resolution
        x_g, y_h = UOS.get_X_Y(slide, x_g, y_h, level)
        x_d, y_b = UOS.get_X_Y(slide, x_d, y_b, level)

        list_roi += ReturnGrid(x_g, x_d, y_h, y_b, fixed_size)
    # pdb.set_trace()
    return list_roi


def ROI(name, ref_level=4, disk_size=4, thresh=None, black_spots=None,
        number_of_pixels_max=1000000, verbose=False, marge=0, method='grid',
        mask_address=None, contour_size=3, N_squares=100, seed=None, 
        fixed_size_in=(512,  512), fixed_size_out=(512, 512), cut_whitescore=0.8,
        ticket_val=80):
    triple_tup = (ticket_val, ticket_val, ticket_val)
    # creates a grid of the all interesting places on the image

    if seed is not None:
        random.seed(seed)

    if '/' in name:
        cut = name.split('/')[-1]
        folder = cut.split('.')[0]
    else:
        folder = name.split(".")[0]
    slide = openslide.open_slide(name)
    list_roi = []
    # pdb.set_trace()

    if method == 'grid':
        lowest_res = len(slide.level_dimensions) - 2

        s = np.array(slide.read_region((0, 0), lowest_res,
                                       slide.level_dimensions[lowest_res]))[:, :, 0:3]

        binary = ROI_binary_mask(s, ticket=triple_tup)
        stru = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        blobs, number_of_blobs = ndimage.label(binary, structure=stru)
        for i in range(1, number_of_blobs + 1):
            y, x = np.where(blobs == i)
            x_0 = min(x)
            y_0 = min(y)
            w = max(x) - x_0
            h = max(y) - y_0
            new_x, new_y = UOS.get_X_Y(slide, x_0, y_0, lowest_res)
            list_roi = Best_Finder_rec(slide, lowest_res, new_x, new_y, w, h, "./" + folder +
                                       "/" + folder, ref_level, list_roi, number_of_pixels_max, thresh, verbose)

    elif method == 'grid_etienne':
        lowest_res = len(slide.level_dimensions) - 2

        s = np.array(slide.read_region((0, 0), lowest_res,
                                       slide.level_dimensions[lowest_res]))[:, :, 0:3]

        binary = ROI_binary_mask(s, ticket=triple_tup)
        stru = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        blobs, number_of_blobs = ndimage.label(binary, structure=stru)
        for i in range(1, number_of_blobs + 1):
            y, x = np.where(blobs == i)
            x_0 = min(x)
            y_0 = min(y)
            w = max(x) - x_0
            h = max(y) - y_0
            new_x, new_y = UOS.get_X_Y(slide, x_0, y_0, lowest_res)
            options = {}
            options['Threshold'] = thresh
            options['value'] = cut_whitescore
            # pdb.set_trace()
            list_roi = Best_Finder_rec(slide, lowest_res, new_x, new_y, w, h,
                                       ref_level, list_roi, number_of_pixels_max, marge, options)

    elif method == 'SP_ROI_normal':

        lowest_res = len(slide.level_dimensions) - 2

        s = np.array(slide.read_region((0, 0), lowest_res,
                                       slide.level_dimensions[lowest_res]))[:, :, 0:3]

        binary = ROI_binary_mask(s, ticket=triple_tup)
        contour_binary = GetContours(binary)

        list_roi = []
        mask = np.zeros(shape=(binary.shape[0], binary.shape[1]), dtype='uint8')

        if isinstance(N_squares, int):
            n_1 = N_squares / 2
            n_2 = n_1
        elif isinstance(N_squares, tuple):
            n_1 = N_squares[0]
            n_2 = N_squares[1]
        else:
            raise NameError("Issue number 0001")
            return []

        list_outside, mask = Sample_imagette(binary, n_1, slide, ref_level,
                                             number_of_pixels_max, lowest_res, mask)
        list_contour_binary, mask = Sample_imagette(contour_binary, n_2, slide, ref_level,
                                                    number_of_pixels_max, lowest_res, mask)
        list_roi = list_outside + list_contour_binary

    elif method == 'SP_ROI_tumor':
        #pdb.set_trace()
        lowest_res = len(slide.level_dimensions) - 3

        s = np.array(slide.read_region((0, 0), lowest_res, slide.level_dimensions[lowest_res]))[:, :, 0:3]
        name_mask = name.replace("/Tumor/", "/Tumor_Mask/").replace(".tif", "_Mask.tif")
        slide_tumor = openslide.open_slide(name_mask)
        TumorLocation = np.array(slide_tumor.read_region((0, 0), lowest_res,
                                       slide_tumor.level_dimensions[lowest_res]))[:, :, 0]

        binary = ROI_binary_mask(s, ticket=triple_tup).astype('uint8')
        contour_binary = GetContours(binary)

        binary[binary > 0] = 255
        TumorLocation[TumorLocation > 0] = 255

        NonTumorButTissue = binary - TumorLocation.copy()
        NonTumorButTissue[NonTumorButTissue < 0] = 0 



        list_roi = []
        mask = np.zeros(shape=(binary.shape[0], binary.shape[1]), dtype='uint8')

        if isinstance(N_squares, int):
            n_1 = N_squares / 2
            n_2 = n_1
        elif isinstance(N_squares, tuple):
            n_1 = N_squares[0]
            n_2 = N_squares[1]
            n_3 = N_squares[2]
        else:
            raise NameError("Issue number 0001")
            return []
        
        list_outside, mask = Sample_imagette(NonTumorButTissue, n_1, slide, ref_level,
                                             number_of_pixels_max, lowest_res, mask)
        list_inside, mask = Sample_imagette(TumorLocation, n_2, slide, ref_level,
                                             number_of_pixels_max, lowest_res, mask)
        list_contour_binary, mask = Sample_imagette(contour_binary, n_3, slide, ref_level,
                                                    number_of_pixels_max, lowest_res, mask)
        list_roi = list_outside + list_inside + list_contour_binary

    elif method == "grid_fixed_size":
        sample = np.array(UOS.GetWholeImage(
            slide, level=(slide.level_count - 2)))[:, :, 0:3]
        res = ROI_binary_mask(sample)
        # pdb.set_trace()
        list_roi = GimmeGrid(slide, res, level=(slide.level_count - 2),
                             fixed_size=fixed_size_in)

    else:
        raise NameError("Not known method")

    list_roi = np.array(list_roi)
    return(list_roi)


def visualise_cut(slide, list_pos, res_to_view=None, color='red', size=12, title=""):
    if res_to_view is None:
        res_to_view = slide.level_count - 3
    whole_slide = np.array(slide.read_region(
        (0, 0), res_to_view, slide.level_dimensions[res_to_view]))
    max_x, max_y = slide.level_dimensions[res_to_view]
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, aspect='equal')
    # ax.imshow(flip_vertical(whole_slide))  # , origin='lower')
    # whole_slide = flip_horizontal(whole_slide)
    ax.imshow(whole_slide)
    for para in list_pos:
        top_left_x, top_left_y = UOS.get_X_Y_from_0(
            slide, para[0], para[1], res_to_view)
        w, h = UOS.get_size(slide, para[2], para[3], para[4], res_to_view)
        p = patches.Rectangle(
            (top_left_x, max_y - top_left_y - h), w, h, fill=False, edgecolor=color)
        p = patches.Rectangle((top_left_x, top_left_y), w,
                              h, fill=False, edgecolor=color)
        ax.add_patch(p)
    ax.set_title(title, size=20)
    plt.show()


# if False:
"""
if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-f", "--file", dest="file",
                      help="Input file (raw data)")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Where to store the patches")
    parser.add_option('-r', '--res', dest="resolution",
                      help="Resolution")
    parser.add_option('--height', dest='h',
                      help="height of an image")
    parser.add_option('--width', dest='w',
                      help="width of an image")
    parser.add_option('--nber_squares', dest='N',
                      help="Number of squares")
    parser.add_option('--perc_contour', dest="perc",
                      help="percentage of squares given to the contours", default="0.2")
    (options, args) = parser.parse_args()

    # checking the input data
    try:
        slide = openslide.open_slide(options.file)
    except:
        print "Issue with file name"
    try:
        # pdb.set_trace()
        if not os.path.isdir(options.output_folder):
            os.mkdir(options.output_folder)
        if _platform == "linux2":
            name = options.file.split('/')[-1].split('.')[0]
            options.output_folder = options.output_folder + "" + name
        else:
            name = options.file.split('\\')[-1].split('.')[0]
            options.output_folder = options.output_folder + "\\" + name
        if not os.path.isdir(options.output_folder):
            os.mkdir(options.output_folder)
    except:
        print "Failed to check output folder..."
    try:
        options.resolution = int(options.resolution)
        options.h = int(options.h)
        options.w = int(options.w)
        options.N = int(options.N)
    except:
        print "Problem while converting to int.."
    try:
        options.perc = float(options.perc)
        n_1 = int(options.N * (1 - options.perc))
        n_2 = options.N - n_1
    except:
        print "Problem while converting to float.."

    print "Input paramters to CuttingPatches:"
    print " \n "
    print "Input file        : | " + options.file
    print "Output folder     : | " + options.output_folder
    print "Resolution        : | " + str(options.resolution)
    print "Height of img     : | " + str(options.h)
    print "Width of img      : | " + str(options.w)
    print "Nber on the inside: | " + str(n_1)
    print "Nber on the contour:| " + str(n_2)

    print ' \n '
    print "Beginning analyse:"

    start_time = time.time()
# Core of the code
########################################

    list_of_para = ROI(options.file, method="SP_ROI", ref_level=options.resolution,
                       N_squares=(n_1, n_2), seed=42, number_of_pixels_max=(options.w, options.h), fixed_size_in=(int(options.h), int(options.w)))
    # pdb.set_trace()
    #visualise_cut(openslide.open_slide(options.file), list_of_para,
    #              res_to_view=4, title="This is the ouput of ROI")
    bar = progressbar.ProgressBar()
    i = 0
    for para in bar(list_of_para):
        sample = UOS.GetImage(options.file, para)
        if _platform == "linux2":
            sample.save(options.output_folder + "/" + str(i) + ".png")
        else:
            sample.save(options.output_folder + "\\" + str(i) + ".png")
        i += 1
########################################

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image:"
    diff_time = diff_time / len(list_of_para)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
"""