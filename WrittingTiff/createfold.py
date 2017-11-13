# -*- coding: utf-8 -*-

#from gi.repository import Vips
import openslide
from UsefulOpenSlide import GetImage
import pdb
from Deprocessing.InOutProcess import Forward, ProcessLoss
from skimage.morphology import dilation, erosion, disk
import time
import numpy as np
from Deprocessing.Morphology import DynamicWatershedAlias
import matplotlib.pylab as plt
from UsefulFunctions.RandomUtils import CheckOrCreate, CleanTemp
from tifffile import imsave
import os
from optparse import OptionParser
from CuttingPatches import ROI
from progressbar import ProgressBar
import glob
import warnings
import caffe
warnings.filterwarnings('ignore')


def options():
    parser = OptionParser()

    parser.add_option('--input', dest="input", type="string",
                      help="Input file")
    parser.add_option('--output', dest="output", type="string", default="random_out.tiff",
                      help="Output file")
    parser.add_option('--size', dest="size", type="int",
                      help="Size of the tiles")
    (options, args) = parser.parse_args()
    return options


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
    input_slide = openslide.open_slide(slide)
    local_dir = slide.split('/')[0:-1]
    local_dir = "/" + os.path.join(*local_dir)
    local_dir = os.path.join(local_dir, "temp_build")
    outputfilename = outputfilename if outputfilename is not None else local_dir
    CheckOrCreate(outputfilename)
    dim1, dim2 = input_slide.dimensions
    #output_slide = Vips.Image.black(dim1, dim2)
    pbar = ProgressBar()
    for param in pbar(table):
        image = np.array(GetImage(input_slide, param))[:, :, :3]
        image = f(image)
        outfile = os.path.join(
            outputfilename, "{}_{}.tiff".format(param[0], param[1]))
        imsave(outfile, image)
    return outputfilename


def WritteTiffFromFiles(input_directory, outputtiff, size_x, size_y):
    files_tiff = glob.glob(os.path.join(input_directory, "*.tiff"))
    img = Vips.Image.black(size_x, size_y)
    pbar = ProgressBar()
    for files in pbar(files_tiff):
        tile = Vips.Image.new_from_file(files,
                                        access=Vips.Access.SEQUENTIAL_UNBUFFERED)
        _x, _y = files.split('.')[0].split('/')[-1].split('_')
        img = img.insert(tile, int(_x), int(_y))
    img.tiffsave(outputtiff, tile=True, pyramid=True, bigtiff=True)


def ProcessOneImage(slide, f, output, options):

    size_images = 224 if options.size is None else options.size
    list_of_para = ROI(slide, method="grid_fixed_size",
                       ref_level=0, seed=42, fixed_size_in=(size_images, size_images))
    size_x, size_y = openslide.open_slide(slide_name).dimensions

    #list_of_para = list_of_para[10:100]

    temp_out = ApplyToSlideWrite(slide, list_of_para, f)

    WritteTiffFromFiles(temp_out, output, size_x, size_y)

    CleanTemp(temp_out)


def GetNet(cn, wd):

    root_directory = wd + "/" + cn + "/"
    if 'FCN' not in cn:
        folder = root_directory + "temp_files/"
        weight = folder + "weights." + cn + "_141549" + ".caffemodel"
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

    opt = options()

    slide_name = opt.input
    out_slide = opt.output

    #import caffe
    #caffe.set_mode_cpu()

    cn_1 = "FCN_0.01_0.99_0.0005"

    wd_1 = "/home/pnaylor/Documents/Experiences/FCN"

    net_1 = GetNet(cn_1, wd_1)

    param = 8

    def pred_f(image, param=param):
        # pdb.set_trace()
        prob_image1, bin_image1 = PredImageFromNet(
            net_1, image, with_depross=True)
        segmentation_mask = DynamicWatershedAlias(prob_image1, param)
        segmentation_mask[segmentation_mask > 0] = 1
        contours = dilation(segmentation_mask, disk(2)) - \
            erosion(segmentation_mask, disk(2))

        x, y = np.where(contours == 1)
        image[x, y] = np.array([0, 0, 0])
        return image

    start_time = time.time()
    diff_time = time.time() - start_time

    ProcessOneImage(slide_name, pred_f, out_slide, opt)

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    # print ' \n '
    # print "Average time per image:"
    #diff_time = diff_time / len(list_of_para)
    # print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60,
    # diff_time % 60)

    from UsefulFunctions.EmailSys import ElaborateEmail
    body = "Writting slide {} took ".format(
        opt.input) + '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
    subject = "Writting file"
    ElaborateEmail(body, subject)
