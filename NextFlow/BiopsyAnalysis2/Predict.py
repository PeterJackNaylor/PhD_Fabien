from optparse import OptionParser
from UsefulFunctions.UsefulOpenSlide import GetImage
from Deprocessing.InOutProcess import Forward, ProcessLoss
import openslide
import caffe
import numpy as np
from skimage.io import imsave


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
    parser.add_option('--trained', dest="wd", type='str', 
                      help="folder where the pretrained networks are")

    (options, args) = parser.parse_args()

    options.param = [options.x, options.y, options.ref_level, options.size_x, options.size_y]


    return options

def GetNet(cn, wd):

    root_directory = wd + "/" + cn + "/"
    if 'FCN' not in cn:
        folder = root_directory + "temp_files/"
        weight = folder + "weights." + cn + "_141549" + ".caffemodel"
        deploy = root_directory + "deploy.prototxt"
    else:
        folder = root_directory + "FCN8/temp_files/"
        weight = folder + "weights." + "FCN8_141549" + ".caffemodel"
        deploy = root_directory + "FCN8/deploy.prototxt"
    net = caffe.Net(deploy, weight, caffe.TRAIN)
    return net



if __name__ == "__main__":
    options = options_min()

    wd = options.wd
    cn_1 = "FCN_0.01_0.99_0.0005"
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
    net_1 = GetNet(cn_1, wd)
    net_2 = GetNet(cn_2, wd)

    slide = options.slide
    para = [options.x, options.y, options.size_x, options.size_y, options.ref_level]
    slide = openslide.open_slide(slide)
    image = np.array(GetImage(slide, para))[:,:,:3]

    net_1.blobs['data'].reshape(1, 3, options.size_x, options.size_y)
    net_2.blobs['data'].reshape(1, 3, options.size_x, options.size_y)

    out_1 = Forward(net_1, image)
    out_2 = Forward(net_2, image)

    prob_1 = ProcessLoss(out_1, method='softmax')
    prob_2 = ProcessLoss(out_2, method='softmax')

    prob = (prob_1 + prob_2) / 2
    prob = prob * 255
    prob = prob.astype(np.uint8)
    imsave(options.output ,prob)

