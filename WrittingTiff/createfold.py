# -*- coding: utf-8 -*-

from gi.repository import Vips
import openslide
from UsefulOpenSlide import GetImage
import pdb
from Deprocessing.InOutProcess import Forward, ProcessLoss
from skimage.morphology import dilation, erosion, disk
import time
import numpy as np
from Deprocessing.Morphology import DynamicWatershedAlias
import matplotlib.pylab as plt


def plot(image):
    plt.imshow(image)
    plt.show()


def ApplyToSlideWrite(slide, table, f, outputfilename=None):
        # Slide is a string of the location of the file
        #  This function applies a function f to the whole slide, this slide is given as input with a table
    # which contains all the patches on which to apply the function.
    # Their is also a optionnal outputfilename

    #  table is a iterable where each element has 5 attributes:
    #   x, y, w, h, res

    input_slide = openslide.open_slide(slide)
    outputfilename = outputfilename if outputfilename is not None else "F_" + slide
    dim1, dim2 = input_slide.dimensions
    #output_slide = Vips.Image.black(dim1, dim2)
    red_channel = Vips.Image.black(dim1, dim2)
    green_channel = Vips.Image.black(dim1, dim2)
    blue_channel = Vips.Image.black(dim1, dim2)

    for i in range(len(table)):
        if i % 10 == 0:
            print "process: {} / {} ".format(i, len(table))
        image = np.array(GetImage(input_slide, table[i]))[:, :, :3]
        image = f(image)

        red_part = Vips.Image.new_from_array(image[:, :, 0].tolist())
        green_part = Vips.Image.new_from_array(image[:, :, 1].tolist())
        blue_part = Vips.Image.new_from_array(image[:, :, 2].tolist())

        red_channel = red_channel.insert(red_part, table[i][0], table[i][1])
        green_channel = green_channel.insert(
            green_part, table[i][0], table[i][1])
        blue_channel = blue_channel.insert(blue_part, table[i][0], table[i][1])
        #output_slide = output_slide.insert(image, table[i][0], table[i][1])
    print "lets join the slides"
    rgb = red_part.bandjoin([green_part, blue_part])
    rgb.write_to_file(outputfilename)


def ApplyToSlideWrite(slide, table, f, outputfilename=None):
        # Slide is a string of the location of the file
        #  This function applies a function f to the whole slide, this slide is given as input with a table
    # which contains all the patches on which to apply the function.
    # Their is also a optionnal outputfilename

    #  table is a iterable where each element has 5 attributes:
    #   x, y, w, h, res
    from scipy.misc import imsave
    input_slide = openslide.open_slide(slide)
    outputfilename = outputfilename if outputfilename is not None else "./temp_build/"
    CheckOrCreate(outputfilename)
    dim1, dim2 = input_slide.dimensions
    #output_slide = Vips.Image.black(dim1, dim2)
    for i in range(len(table)):
        if i % 10 == 0:
            print "process: {} / {}     \r".format(i, len(table))
        image = np.array(GetImage(input_slide, table[i]))[:, :, :3]
        image = f(image)
        outfile = os.path.join(
            outputfilename, "{}_{}.tiff".format(table[i][0], table[i][1]))
        tifffile.imsave(outfile, image)
        imsave(
            , image)
    print "lets join the slides"
#    rgb = red_part.bandjoin([green_part, blue_part])
#    rgb.write_to_file(outputfilename)
from tifffile import imsave
from scipy.misc import imread
import sys
import os

for i in range(2, len(sys.argv)):
    name_tiff = sys.argv[i].replace('.jpg', '.tiff')
    to_replace = os.path.join(*sys.argv[i].split('/')[0:-1])
    name_tiff = name_tiff.replace(to_replace, sys.argv[1])

    xx = imread(sys.argv[i])
    imsave(name_tiff, xx)


def GetNet(cn, wd):

    root_directory = wd + "/" + cn + "/"
    if 'FCN' not in cn:
        folder = root_directory + "temp_files/"
        weight = folder + "weights." + cn + ".caffemodel"
        deploy = root_directory + "test.prototxt"
    else:
        folder = root_directory + "FCN8/temp_files/"
        weight = folder + "weights." + "FCN8_141549" + ".caffemodel"
        deploy = root_directory + "FCN8/test.prototxt"

    net = caffe.Net(deploy, weight, caffe.TRAIN)
    return net


def PredImageFromNet(net, image, with_depross=True):
    new_score = Forward(net, image, preprocess=with_depross, layer=["score"])
    bin_map = ProcessLoss(new_score["score"], method="binary")
    prob_map = ProcessLoss(new_score["score"], method="softmax")
    return prob_map, bin_map


if __name__ == '__main__':
    print "In this script, we will take one slide and create a new slide, this new slide will be annotated with cells"

    from CuttingPatches import ROI
    slide_name = "/home/pnaylor/Documents/temp_image/576041.tiff"
    out_slide = "/home/pnaylor/Documents/temp_image/576041_Out.tiff"
    param = 5
    size_images = 224
    list_of_para = ROI(slide_name, method="grid_fixed_size",
                       ref_level=0, seed=42, fixed_size_in=(size_images, size_images))
    i = 0
    import caffe
    caffe.set_mode_cpu()

    cn_1 = "FCN_0.01_0.99_0.0005"
    cn_2 = "batchLAYER4"

    wd_1 = "/home/pnaylor/Documents/Experiences/FCN"
    wd_2 = wd_1

    net_1 = GetNet(cn_1, wd_1)
    #net_2 = GetNet(cn_2, wd_2)

    def predict_ensemble(image):
        prob_image1, bin_image1 = PredImageFromNet(
            net_1, image, with_depross=True)
        # prob_image2, bin_image2 = PredImageFromNet(
        #    net_2, image, with_depross=True)
        #prob_ensemble = (prob_image1 + prob_image2) / 2
        #bin_ensemble = (prob_ensemble > 0.5) + 0
        segmentation_mask = DynamicWatershedAlias(prob_image1, param)
        segmentation_mask[segmentation_mask > 0] = 1
        contours = dilation(segmentation_mask, disk(2)) - \
            erosion(segmentation_mask, disk(2))

        x, y = np.where(contours == 1)
        image[x, y, 0] = 0
        image[x, y, 1] = 0
        image[x, y, 2] = 0

        return image
    start_time = time.time()
    ApplyToSlideWrite(slide_name, list_of_para,
                      predict_ensemble, outputfilename=out_slide)
    diff_time = time.time() - start_time
    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image:"
    diff_time = diff_time / len(list_of_para)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    from UsefulFunctions.EmailSys import ElaborateEmail
    body = "I have finished writting all your bloody damn files so please test out what you wanted to do !!! \n Jerk"
    subject = "Vips step"
    ElaborateEmail(body, subject)
