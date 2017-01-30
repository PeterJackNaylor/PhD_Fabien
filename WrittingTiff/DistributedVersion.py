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

def indent(text, amount=4, ch=' '):
    """ Creates an indented string with amount characters of type ch.
    Default:
    - amount = 4
    - ch = ' ' 
    """
    padding = amount * ch
    return ('\n'+padding+text)


def CreateFileParam(name, list):
    """
    Creates physically a text file named name where each line as an id 
    and each line as parameters
    """
    f = open(name, "wb")
    line = 1
    for para in list:
        pre = "__{}__ ".format(line)
        pre += "{} {} {} {} {}\n".format(*para)
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
               verbose=False, marge=0, method='grid_etienne', 
               mask_address=None, contour_size=3, N_squares=100, 
               seed=None, fixed_size_in=(512, 512), fixed_size_out=(512,512))
    distribute_file = os.path.join(output, "ParameterDistribution.txt")
    bash_file = os.path.join(output, "PredOneSlide.sh")
    python_file = os.path.join(output, "PredictionSlide.py")
    CheckOrCreate(output)
    CreateFileParam(distribute_file, list_of_para)
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
           options: dictionnary of option such as output, slide name, size, and options names for the options to the python script
           
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
    n_field = len(open(file_param, "rb").readlines()[0].split(' ')) - 1
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

stepSize = 200
windowSize = 224
param = 8

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, x + windowSize[0], y + windowSize[1], image[y:y + windowSize[1], x:x + windowSize[0]])


def PredLargeImageFromNet(net_1, image, stepSize, windowSize):
    pdb.set_trace() 
    x_s, y_s, z_s = image.shape
    result = np.zeros(shape=(x_s, y_s, 2))
    for x_b, y_b, x_e, y_e, window in sliding_window(image, stepSize, windowSize):
        prob_image1, bin_image1 = PredImageFromNet(net_1, image, with_depross=True)
        result[y_b:y_e, x_b:x_e,  0] += prob_image1
        result[y_b:y_e, x_b:x_e,  1] += 1
    prob_map = result[:,:,0] / result[:,:,1]
    bin_map = prob_map > 0.5 + 0.0
    return prob_map, bin_map

def pred_f(image, stepSize=stepSize, windowSize=windowSize, param=param):
    pdb.set_trace()
    caffe.set_mode_cpu()
    cn_1 = "FCN_0.01_0.99_0.0005"
    wd_1 = "/share/data40T_v2/Peter/pretrained_models"
    net_1 = GetNet(cn_1, wd_1)
    prob_image1, bin_image1 = PredLargeImageFromNet(net_1, image, stepSize, windowSize)
    #pdb.set_trace()
    segmentation_mask = DynamicWatershedAlias(prob_image1, param)
    segmentation_mask[segmentation_mask > 0] = 1
    contours = dilation(segmentation_mask, disk(2)) - \
        erosion(segmentation_mask, disk(2))

    x, y = np.where(contours == 1)
    image[x, y] = np.array([0, 0, 0])
    return image

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
                      help="Method of the tilling procedure")
    parser.add_option('--tc', dest="tc", type="int", default=50,
                      help="Numbr of jobs in paralelle")
    (options, args) = parser.parse_args()
    options.name = ["-x", "-y", "--size_x", "--size_y", "--ref_level"]
    return options


def PredImage(options):
    slide = options.slide
    para = [options.x, options.y, options.size_x, options.size_y, options.ref_level]
    outfile = os.path.join(options.output, 'tiled',
                           "{}_{}.tiff".format(options.x, options.y))
    print "slide :{}".format(slide)
    print "para : ", para
    print "outfile : {}".format(outfile)
    print "f : ", options.f
    PredOneImage(slide, para, outfile, options.f)

def PredOneImage(slide, para, outfile, f):
    # pdb.set_trace()
    slide = openslide.open_slide(slide)
    image = np.array(GetImage(slide, para))[:,:,:3]
    pdb.set_trace()
    image = f(image)
    imsave(outfile, image, resolution=[1.0,1.0])

if __name__ == "__main__":

    options = options_all()
    Distribute(options.slide, options.size, options.output, options)
