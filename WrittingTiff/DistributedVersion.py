from optparse import OptionParser
import pdb
from UsefulFunctions.UsefulOpenSlide import GetImage
from UsefulFunctions.RandomUtils import CheckOrCreate
from CuttingPatches import ROI
from tifffile import imsave
import numpy as np
import os
from UsefulFunctions.RandomUtils import CheckOrCreate, CleanTemp
import time
import getpass
import openslide
from Extractors import bin_analyser, list_f
from skimage import img_as_ubyte

def indent(text, amount=4, ch=' '):
    """ Creates an indented string with amount characters of type ch.
    Default:
    - amount = 4
    - ch = ' ' 
    """
    padding = amount * ch
    return ('\n'+padding+text)


def CreateFileParam(name, list, slidename):
    """
    Creates physically a text file named name where each line as an id 
    and each line as parameters
    """
    f = open(name, "wb")
    line = 1
    for para in list:
        pre = "__{}__ ".format(line)
        pre += "{} {} {} {} {}".format(*para)
        pre += " {}".format(slidename)
        pre += "\n"
        f.write(pre)
        line += 1
    f.close()


def Distribute(slide, size, output, options):
    """
    Input:
    Slide : name of the slide to analyse
    size: size of outputed image crops from size.
    output: folder to store all the different files for the processing
    options: dictionnary with many options.

    Ouput: No output but a folder in output where you will find
        - a parameter file called ParameterDistribution
        - a bash file called PredOneSlide.sh
        - a python file called PredictionSlide.py
    """
    # the size option doesn't matter here...
    #list_of_para = ROI(slide, method=options.method,
    #                   ref_level=0, seed=42, fixed_size_in=(size, size))
    list_of_para = ROI(slide, ref_level=0, disk_size=4, thresh=230, 
               black_spots=None, number_of_pixels_max=9000000, 
               verbose=False, marge=options.marge, method=options.method, 
               mask_address=None, contour_size=3, N_squares=100, 
               seed=None, fixed_size_in=(512, 512), fixed_size_out=(512,512))
    distribute_file = os.path.join(output, "ParameterDistribution.txt")
    bash_file = os.path.join(output, "PredOneSlide.sh")
    python_file = os.path.join(output, "PredictionSlide.py")
    PBS = os.path.join(output, "PBS")
    OUT = os.path.join(PBS, "OUT")
    ERR = os.path.join(PBS, "ERR")
    tiled = os.path.join(output, "tiled")
    table = os.path.join(output, "table")
    CheckOrCreate(output)
    CheckOrCreate(PBS)
    CheckOrCreate(OUT)
    CheckOrCreate(ERR)
    CheckOrCreate(tiled)
    CheckOrCreate(table)
    CreateFileParam(distribute_file, list_of_para, os.path.basename(slide).replace(".tiff", ""))
    CreateBash(bash_file, python_file, distribute_file, options)
    CreatePython(python_file, options)


def CreatePython(python_file, options):
    """
    Creates a small python file which is submitted to a cluster of jobs.
    input:
        python file: name of file
        options: dictionnary of options such as: None are needed here.
    """
    f = open(python_file, "wb")
    f.write("# -*- coding: cp1252 -*- \n\"\"\" \nDescription: \n \nAuthors: " +
            getpass.getuser() + "\n \nDate: " + time.strftime("%x") + "\n \n\"\"\" ")
    f.write(" \n \nimport WrittingTiff.DistributedVersion\n \nif __name__ ==  \"__main__\": \n \n")
    f.write(indent("options = WrittingTiff.DistributedVersion.options_min()\n "))
    f.write(indent("WrittingTiff.DistributedVersion.PredImage(options)"))
    f.write("\n\n" + "#" * 100 + "\n \n")
    f.close()


def CreateBash(bash_file, python_file, file_param, options):
    """
    Creates a bash function to submit in order to analyse each image patches.
    input: bash_file: name of the bash file
           python_file: name of the python file to submit.
           file_param: name of the parameter file.
           options: dictionnary of option such as output,
           slide name, size, and options names for the options to the python script
           
    """
    f = open(bash_file, "wb")
    f.write("#!/bin/bash\n\n")  # sets bash environnement
    f.write("#$ -cwd\n")  # executes job in current directory
    f.write("#$ -S /bin/bash\n")  # set bash environment
    # name of the job as it will appear in qstat -f
    num = options.slide.split('/')[-1].split('.')[0]
    f.write("#$ -N ProcessSlide{} \n".format(num))
    OUT_PBS = os.path.join(options.output, "PBS", "OUT")
    ERR_PBS = os.path.join(options.output, "PBS", "ERR")
    f.write("#$ -o " + OUT_PBS + "\n")  # where to put the output "print"
    f.write("#$ -e " + ERR_PBS + "\n")  # where to put the error messages
    n = len(open(file_param, "rb").readlines())
    f.write("#$ -t 1-{}\n".format(n))
    n_tc = 50 if options.tc is None else options.tc
    f.write("#$ -tc {}".format(n_tc))

    f.write('\n\n\n')
    f.write('PYTHON_FILE={}\n'.format(python_file))
    f.write('FILE={}\n'.format(file_param))
    f.write('OUTPUT={}\n'.format(options.output))
    f.write('SLIDE={}\n'.format(options.slide))
    f.write('SIZE={}\n'.format(options.size))
    f.write('spe_tag=__\n')
    last_line = ""
    n_field = len(open(file_param, "rb").readlines()[0].split(' ')) - 2
    for i in range(n_field):
        f.write(
            'FIELD{}=$(grep \"$spe_tag$SGE_TASK_ID$spe_tag \" $FILE | cut -d\' \' -f{})\n'.format(i, i + 2))
        last_line += "{} $FIELD{} ".format(options.name[i], i)
    last_line += "--output $OUTPUT "
    last_line += "--slide $SLIDE "
    last_line += "--size $SIZE"
    f.write("\n\npython " + "$PYTHON_FILE" + " " + last_line)
    f.write("\n\n" + "#" * 100 + "\n \n")
    f.close()


###################
import caffe
from createfold import GetNet, PredImageFromNet, DynamicWatershedAlias, dilation, disk, erosion
from UsefulFunctions.UsefulImageConstruction import sliding_window, PredLargeImageFromNet
from skimage.morphology import remove_small_objects
import skimage as skm
from scipy import misc
import cv2
from Extractors import *


stepSize = 175
windowSize = (224 , 224)
param = 10


def pred_image_from_two_nets(image, net1, net2, stepSize, windowSize,
                             param=param, marge=1, method="avg", ClearBorder = "RemoveBorderWithDWS"):
    prob_image1, bin_image1, threshold1 = PredLargeImageFromNet(net1, image, stepSize, windowSize,
                                                                removeFromBorder=marge,
                                                                method=method,
                                                                param=param,
                                                                ClearBorder=ClearBorder)
    prob_image2, bin_image2, threshold2 = PredLargeImageFromNet(net2, image, stepSize, windowSize,
                                                                removeFromBorder=marge,
                                                                method=method,
                                                                param=param,
                                                                ClearBorder=ClearBorder)
    
    thresh = ( threshold1 + threshold2 ) / 2.
    prob = ( prob_image1 + prob_image2 ) / 2.
    bin_map = prob > thresh + 0.0
    bin_map = bin_map.astype(np.uint8)
    return prob, bin, thresh



def pred_f(image, stepSize=stepSize, windowSize=windowSize, param=param, 
           marge=None, marge_cut_off=0, ClearSmallObjects=20, list_f=list_f):
    caffe.set_mode_cpu()
    cn_1 = "FCN_0.01_0.99_0.0005"
    wd_1 = "/share/data40T_v2/Peter/pretrained_models"
    net_1 = GetNet(cn_1, wd_1)
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
    net_2 = GetNet(cn_2, wd_1)
    prob_image, bin_image, thresh = pred_image_from_two_nets(image, net_1, net_2, stepSize, windowSize, 
                                                               param=param, marge=marge, method="avg", 
                                                               ClearBorder="Reconstruction")

    segmentation_mask = DynamicWatershedAlias(prob_image, param)
    segmentation_mask = remove_small_objects(segmentation_mask, ClearSmallObjects)
    table = bin_analyser(image, segmentation_mask, list_f, marge_cut_off)
    segmentation_mask[segmentation_mask > 0] = 1.


    contours = dilation(segmentation_mask, disk(2)) - \
        erosion(segmentation_mask, disk(2))
    x, y = np.where(contours == 1)
    image[x, y] = np.array([0, 0, 0])


    segmentation_mask = img_as_ubyte(segmentation_mask)
    segmentation_mask[segmentation_mask > 0] = 255
    if marge_cut_off != 0:
         c = marge_cut_off
         image = image[c:-c, c:-c]
         segmentation_mask = segmentation_mask[c:-c, c:-c]
         prob_image = prob_image[c:-c, c:-c]
    return image, table, segmentation_mask, prob_image


##########################


def options_min():

    parser = OptionParser()

    parser.add_option('--slide', dest="slide", type="string",
                      help="Input slide")
    parser.add_option('-x', dest="x", type="int",
                      help="position on x axis")
    parser.add_option('-y', dest="y", type="int",
                      help="position on y axis")
    parser.add_option('--size_x', dest="size_x", type="int",
                      help='Size of images x axis')
    parser.add_option('--size_y', dest="size_y", type="int",
                      help='Size of images y axis')  
    parser.add_option('--ref_level', dest="ref_level", type ="int",
                       help="Level of resolution")  
    parser.add_option('--output', dest='output', type="str",
		              help='Output folder')
    parser.add_option('--size', dest="size", type="int",
                      help="Size of prediction images")
    parser.add_option('--marge', dest="marge", type='int', default=0,
                      help="Margin for the reconstruction")
    parser.add_option('--marge_cut_off', dest="marge_cut_off", type='int', default=0,
                      help="margin to cut off at the end of the analyse")
    (options, args) = parser.parse_args()

    options.param = [options.x, options.y, options.ref_level, options.size_x, options.size_y]

    options.f = pred_f

    return options


def options_all():

    parser = OptionParser()

    parser.add_option('--slide', dest="slide", type="string",
                      help="Input slide")
    parser.add_option('--output', dest="output", type="string",
                      help="Output folder")
    parser.add_option('--size_tiles', dest="size", type="int",
                      help="Size of the tiles")
    parser.add_option('--method', dest="method", type="str", default='grid_fixed_size',
                      help="Method of the tilling procedure, it can grid_etienne/grid_fixed_size")
    parser.add_option('--tc', dest="tc", type="int", default=50,
                      help="Number of jobs in paralelle")
    parser.add_option('--marge', dest="marge", type='int', default=0,
                      help="Margin for the reconstruction")
    (options, args) = parser.parse_args()
    options.name = ["-x", "-y", "--size_x", "--size_y", "--ref_level"]
    return options


def PredImage(options):
    slide = options.slide
    para = [options.x, options.y, options.size_x, options.size_y, options.ref_level]
    slide_num = options.slide.split('.')[0].split('/')[-1]
    outfile = os.path.join(options.output, 'tiled',
                           "{}_{}_{}_{}_{}_{}.tiff".format(slide_num, options.x, options.y,
                           options.size_x, options.size_y, options.ref_level))

    CheckOrCreate(os.path.join(options.output, "tiled"))
    CheckOrCreate(os.path.join(options.output, "table"))
    CheckOrCreate(os.path.join(options.output, "bin"))
    CheckOrCreate(os.path.join(options.output, "prob"))
    print "slide :{}".format(slide)
    print "para : ", para
    print "outfile : {}".format(outfile)
    print "f : ", options.f
    PredOneImage(slide, para, outfile, options.f, options)


from tifffile import imsave

def PredOneImage(slide, para, outfile, f, options):
    # pdb.set_trace()
    slide = openslide.open_slide(slide)
    image = np.array(GetImage(slide, para))[:,:,:3]
    image, table, bin, prob = f(image, marge=options.marge, marge_cut_off=options.marge_cut_off)
    imsave(outfile, image, resolution=[1.0,1.0])
    np.save(outfile.replace('.tiff', ".npy").replace("tiled", "table"), table)
    imsave(outfile.replace("tiled", "bin"), bin, resolution=[1.0,1.0])
    imsave(outfile.replace("tiled", "prob"), img_as_ubyte(prob), resolution=[1.0,1.0])

def CheckJob(parameter_file, output_folder):
    f = open(parameter_file, "r")
    lines = [ss.split(' ') for ss in f.readlines()]
    for para in lines:
        line_number = para[0]
        x = para[1]
        y = para[2]
        size_x = para[3]
        size_y = para[4]
        ref_level = para[5][0]
        filename = os.path.join(output_folder, "{}_{}_{}_{}_{}.tiff")
        filename = filename.format(x, y, size_x, size_y, ref_level)
        if not os.path.isfile(filename):
            print "file {} doesn't exists whish is job array number {}".format(filename, line_number)

if __name__ == "__main__":

    options = options_all()
    Distribute(options.slide, options.size, options.output, options)
