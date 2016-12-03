
from UsefulFunctions.UsefulOpenSlide import GetImage
from UsefulFunctions.RandomUtils import CheckOrCreate
from CuttingPatches import ROI
from tifffile import imsave
import numpy as np


def CreateFileParam(name, list):
	f = open(os.path.join(INP_folder, "parameters.txt"), "wb")
	line = 1
	for para in list:
		small_para = [para[0], para[1], para[3]]
		pre = "__{}__ ".format(line)
		pre += "{} {} {}".format(*small_para)
        line += 1
	f.close()


def Distribute(slide, size, output, options):
	list_of_para = ROI(slide, method=options.method,
                       ref_level=0, seed=42, fixed_size_in=(size, size))
	distribute_file = os.path.join(output, "ParameterDistribution.txt")
    bash_file = os.path.join(output, "PredOneSlide.sh")
	CreateFile(distribute_file, list_of_para)
	CreateBash(bash_files, distribute_file, options)
    CreatePython(python_file,)


def CreatePython(python_file, options):
    f = open(python_file , "wb" )
    f.write("# -*- coding: cp1252 -*- \n\"\"\" \nDescription: \n \nAuthors: " + getpass.getuser() + "\n \nDate: " + time.strftime("%x") + "\n \n\"\"\" ")
    f.write(" \n \nfrom WrittingTiff.DistributedVersion\n \nif __name__ ==  \"__main__\": \n \n")
    f.write( indent("options = WrittingTiff.DistributedVersion.options_min()\n ") )
    f.write( indent("WrittingTiff.DistributedVersion.PredImage(options)"))
                f.write( "\n\n"+"#" * 100 + "\n \n")
                f.close()


def CreateBash(bash_file, python_file, file_param, options):
    f = open(bash_file, "wb")
    f.write("#!/bin/bash\n\n")  # sets bash environnement
    f.write("#$ -cwd\n")  # executes job in current directory
    f.write("#$ -S /bin/bash\n")  # set bash environment
    # name of the job as it will appear in qstat -f
    num = options.slide.split('/')[-1].split('.')[0]
    f.write("#$ -N ProcessSlide{} \n".num)
    OUT_PBS = os.path.join(options.output, "PBS", "OUT")
    ERR_PBS = os.path.join(options.output, "PBS", "ERR")
    f.write("#$ -o " + OUT_PBS + "\n")  # where to put the output "print"
    f.write("#$ -e " + ERR_PBS + "\n")  # where to put the error messages
    n = len(open(file_param, "rb").readlines())
    f.write("#$ -t 1-{}".format(n))
    n_tc = 50 if options.tc is None else options.tc
    f.write("#$ -tc {}".format(n_tc))

    f.write('\n\n\n')
    f.write('PYTHON_FILE={}\n'.format(python_file))
    f.write('FILE={}\n'.format(file_param))
    f.write('OUTPUT={}\n'.format(options.out))
    f.write('spe_tag=__\n')
    line = 0
    last_line = ""
    n_field = len(open(file_param, "rb").readlines()[0].split(' ')) - 2
    for i in range(n_field):
        f.write(
            'FIELD{}=$(grep \"$spe_tag$SGE_TASK_ID$spe_tag \" $FILE | cut -d\' \' -f{})\n'.format(i, i + 2))
        last_line += "{} $FIELD{} ".format(options.name[i], i)

    f.write("\n\npython " + "$PYTHON_FILE" + " " + last_line)
    f.write("\n\n" + "#" * 100 + "\n \n")
    f.close()


###################
from createfold import caffe, GetNet, PredImageFromNet, DynamicWatershedAlias, dilation, disk, erosion

param = 8
def pred_f(image, param=param):
    # pdb.set_trace()
    caffe.set_mode_cpu()
    cn_1 = "FCN_0.01_0.99_0.0005"
    wd_1 = "/home/pnaylor/Documents/Experiences/FCN"
    net_1 = GetNet(cn_1, wd_1)
    prob_image1, bin_image1 = PredImageFromNet(
        net_1, image, with_depross=True)
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
    parser.add_option('-s', '--size', dest="size", type="int",
    				 help='Size of images')
    (options, args)=parser.parse_args()

    options.param=[options.x, options.y, 0, options.size, options.size]

    options.f = pred_f

    return options


def options_all():

    parser=OptionParser()

    parser.add_option('--slide', dest="slide", type="string",
                      help="Input slide")
    parser.add_option('--output', dest="output", type="string",
                      help="Output folder")
    parser.add_option('--size', dest="size", type="int",
                      help="Size of the tiles")
    parser.add_option('--method', dest="method", type="str", default='grid_fixed_size',
                      help="Method of the tilling procedure")

    (options, args)=parser.parse_args()
    options.name=["-x", "-y", "-s"]
    return options

def PredImage(options):
    slide = options.slide
    para = [options.x, options.y, 0, options.s, options.s]
    outfile = os.path.join(options.outfile, 'tiled', "{}_{}.tiff".format(options.x, options.y))



def PredOneImage(slide, para, outfile, f):
	image=GetImage(slide, para)
	image=f(image)
	imsave(outfile, image)

if __name__="__main__":

	options=options_all()
    Distribute(options.slide, options.size, options.output, options)

