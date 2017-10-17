from optparse import OptionParser
from UsefulFunctions.UsefulOpenSlide import GetImage
#from Deprocessing.InOutProcess import Forward, ProcessLoss
import openslide
import caffe
import numpy as np
from tifffile import imsave
from UsefulFunctions.UsefulImageConstruction import sliding_window, PredLargeImageFromNet


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
#    net_1 = GetNet(cn_1, wd)
#    net_2 = GetNet(cn_2, wd)

    slide = options.slide
    para = [options.x, options.y, options.size_x, options.size_y, options.ref_level]
    slide = openslide.open_slide(slide)
    image = np.array(GetImage(slide, para))[:,:,:3]


    windowSize = min(options.size_x, options.size_y) / 2
    windowSize = (windowSize, windowSize)
    stepSize = windowSize[0] - 50
    prob_fin = np.zeros(shape=(image.shape[0], image.shape[1])).astype(float)

    for cn in [cn_1, cn_2]:
        net = GetNet(cn, wd)
        net.blobs['data'].reshape(1, 3, windowSize[0], windowSize[1])
        prob, bin, thresh = PredLargeImageFromNet(net, image, stepSize, windowSize, 1, 'avg', 7, "RemoveBorderWithDWS", 0.5)
        prob_fin += prob
        del net

    prob = prob_fin / 2
    prob = prob * 255
    prob = prob.astype(np.uint8)
    imsave(options.output, prob, resolution=[1.0,1.0])
    imsave(options.output.replace('prob', 'rgb'), image, resolution=[1.0,1.0])

