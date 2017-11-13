from optparse import OptionParser
from UsefulFunctions.UsefulOpenSlide import GetImage
#from Deprocessing.InOutProcess import Forward, ProcessLoss
import openslide
import caffe
caffe.set_mode_gpu()
import numpy as np
from tifffile import imsave, imread
from UsefulFunctions.UsefulImageConstruction import sliding_window, PredLargeImageFromNet
import tempfile
import shutil
import os.path
from os.path import basename, isfile, join
import pdb


PROB = tempfile.mkdtemp()

def options_min():

    parser = OptionParser()

    parser.add_option('--slide', dest="slide", type="string",
                      help="Input slide")
    parser.add_option('--parameter', dest="parameter", type="str",
                      help="file of whole parameters") 
    parser.add_option('--output', dest='output', type="str",
                      help='Output folder')
    parser.add_option('--trained', dest="wd", type='str', 
                      help="folder where the pretrained networks are")

    (options, args) = parser.parse_args()

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

def LoadFile(slide, para, dir_tmp):
    fname = "rgb_{}_{}_{}_{}_{}_{}.tiff"
    param_ = [basename(slide).split('.')[0]] + para
    fname = fname.format(*param_)
    fname = join(dir_tmp, fname)
    if isfile(fname):
        rgb = imread(fname)
    else:
        rgb = np.array(GetImage(slide, para))[:,:,:3]
        imsave(fname, rgb)
    return rgb

def SaveProb(slide, para, dir_tmp, prob):
    fname = "prob_{}_{}_{}_{}_{}_{}.tiff"
    param_ = [basename(slide).split('.')[0]] + para
    fname = fname.format(*param_)
    tmp_fname = join(dir_tmp, fname)

    if isfile(tmp_fname):
        oldprob = imread(tmp_fname)
        oldprob = oldprob.astype(float)
        oldprob = oldprob / 255.
        oldprob += prob 
        oldprob = oldprob / 2
        oldprob = oldprob * 255.
        oldprob = oldprob.astype(np.uint8)
        imsave(join('.', fname), oldprob, resolution=[1.0,1.0])
    else:
        prob = prob * 255
        prob = prob.astype(np.uint8)
        imsave(tmp_fname, prob, resolution=[1.0,1.0])

if __name__ == "__main__":
    options = options_min()

    wd = options.wd
    cn_1 = "FCN_0.01_0.99_0.0005"
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
#    net_1 = GetNet(cn_1, wd)
#    net_2 = GetNet(cn_2, wd)

    slide = options.slide
    slide = openslide.open_slide(slide)
    file_content = open(options.parameter)
    file_content = file_content.readlines()



    for cn in [cn_1, cn_2]:
        net = GetNet(cn, wd)
        for lines in file_content:
            p = lines.split(' ')
            para = [int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])]
            if cn == cn_1:
                windowSize_x = min(min(p[2], p[3]), 800)
            else:
                windowSize_x = min(min(p[2], p[3]), 500)
            windowSize = (windowSize_x, windowSize_x)
            stepSize = windowSize_x - 50
            net.blobs['data'].reshape(1, 3, windowSize_x, windowSize_x)
            image = LoadFile(options.slide, para, ".")
            prob, bin, thresh = PredLargeImageFromNet(net, image, stepSize, windowSize, 1, 'avg', 7, "RemoveBorderWithDWS", 0.5)
            SaveProb(options.slide, para, PROB, prob)
        del net
    shutil.rmtree(PROB)
