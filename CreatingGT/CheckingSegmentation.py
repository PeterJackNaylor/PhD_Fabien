"""
This script will merge pictures and the segmentation test given by
cell cognition, the format of the folders/names of files will be the
exact same as the one's originaly given by cell cognition
"""

import os
import glob
from skimage.io import imread as ir
from optparse import OptionParser
import time
from usefulPloting import Contours, ImageSegmentationSave


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def CheckExistants(path):
    assert os.path.isdir(path)


"""
We will take the name from the input directory, transform it into
the name and appended it to the label path name
"""


def GetNumber(filename):
    if '/' in filename:
        filename = filename.split('/')[-1]
    if '.' in filename:
        filename = filename.split('.')[0]
    return int(filename)


def CellCecognitionName(path, number):

    name = "P00{:03d}_01_T00000.tif".format(number)

    return os.path.join(path, name)


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-o", "--output", dest="output",
                      help="Output folder")
    parser.add_option("-i", "--input", dest="input",
                      help="Input folder")
    parser.add_option('--label_h', dest="label_h",
                      help="label for histonucgrey (cellcognition folder")
    parser.add_option('--label_e', dest='label_e',
                      help="label for expanded (cellcognition folder)")
    (options, args) = parser.parse_args()

    # checking the input data

    CheckOrCreate(options.output)
    label_h_output = os.path.join(options.output, "label_h")
    label_e_output = os.path.join(options.output, "label_e")

    CheckOrCreate(label_e_output)
    CheckOrCreate(label_h_output)

    CheckExistants(options.input)

    if options.input[-1] != "/":
        options.input += "/"

    if options.label_h is None:
        assert options.label_e is not None
        CheckExistants(options.label_e)
    elif options.label_e is None:
        assert options.label_h is not None
        CheckExistants(options.label_h)
    else:
        CheckExistants(options.label_h)
        CheckExistants(options.label_e)

    print "Input paramters to CheckingSegmentation:"
    print " \n "
    print "Input file        : | " + options.input
    print "Output folder     : | " + options.output
    print "label_h           : | " + options.label_h
    print "label_e           : | " + options.label_e

    print ' \n '
    print "Beginning analyse:"

    start_time = time.time()
# Core of the code
########################################
    output = {options.label_h: label_h_output,
              options.label_e: label_e_output}
    images_names = glob.glob(options.input + "*.png")

    for png in images_names:
        RGB = ir(png)
        number = GetNumber(png)
        for output_path in [options.label_h, options.label_e]:
            if output_path is not None:
                _name = CellCecognitionName(output_path, number)
                _segmentation = ir(_name) > 0
                _segmentation = _segmentation + 0
                savename = os.path.join(output[output_path], str(number))
                ImageSegmentationSave(RGB, _segmentation.transpose(), savename)


########################################

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for all:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image:"
    diff_time = diff_time / (2 * len(images_names))
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
