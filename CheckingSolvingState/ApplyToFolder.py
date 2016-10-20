# -*- coding: utf-8 -*-

import os
import glob
from skimage.io import imread as ir
from skimage.io import imsave
from Deprocessing.Morphology import GetContours, DynamicWatershedAlias

from optparse import OptionParser
import time
from usefulPloting import Contours, ImageSegmentationSave
import caffe
import numpy as np
from ShortPrediction import Preprocessing, OutputNet

#  predicting from ensemble/ dictionnary of nets


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)


def pred_img(image, net_dic, return_prob=False, return_both=False):
    preproc = Preprocessing(image)
    pred_prob = {}
    for name, net in net_dic.items():
        net.blobs['data'].data[0] = preproc
        conv1_name = [el for el in net.blobs.keys() if "conv" in el][0]
        new_score = net.forward(["data"], start=conv1_name, end='score')
        pred_prob[name] = OutputNet(new_score["score"], method='softmax')
    n = len(net_dic)
    res = np.zeros_like(pred_prob[name])
    for name, prob_ in pred_prob.items():
        res += prob_
    bin_res = (((res / n) > 0.5) + 0)
    if return_prob:
        return res / n
    elif return_both:
        return bin_res, res / n
    else:
        return bin_res


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-o", "--output", dest="output",
                      help="Output folder")
    parser.add_option("-i", "--input", dest="input",
                      help="Input folder")
    parser.add_option("--lamb", dest="lamb",
                      help="lambda for hole filling")
    (options, args) = parser.parse_args()

    # checking the input data
    caffe.set_mode_cpu()

    CheckOrCreate(options.output)
    CheckExistants(options.input)

    deploy1 = "/data/users/pnaylor/Documents/Python/newdatagenAll/FCN_randomcrop/FCN8/test.prototxt"
    weight1 = "/data/users/pnaylor/Documents/Python/newdatagenAll/FCN_randomcrop/FCN8/temp_files/weights.FCN8.caffemodel"
    deploy2 = "/data/users/pnaylor/Documents/Python/newdatagenAll/batchLAYER4/test.prototxt"
    weight2 = "/data/users/pnaylor/Documents/Python/newdatagenAll/batchLAYER4/temp_files/weights.batchLAYER4.caffemodel"
    net1 = caffe.Net(deploy1,
                     weight1,
                     caffe.TEST)

    net2 = caffe.Net(deploy2,
                     weight2,
                     caffe.TEST)
    net_dic = {'FCN_randomcrop': net1, 'batchLAYER4': net2}

    print "Input paramters to OutputNet:"
    print " \n "
    print "Input file        : | " + options.input
    print "Output folder     : | " + options.output
    print "lambda            : | " + options.lamb
    print ' \n '
    print "Beginning analyse:"

    start_time = time.time()
    files = glob.glob(os.path.join(options.input, "*.png"))
    for f in files:
        image = ir(f)[0:224,0:224,0:3]
        bin, prob = pred_img(
            image, net_dic, return_prob=False, return_both=True)
        dyn_ws = DynamicWatershedAlias(prob, int(options.lamb))
        dyn_ws_contours = GetContours(dyn_ws)
        normal_contours = GetContours(bin)

        Overlay_normal = image.copy()
        Overlay_dyn_ws = image.copy()

        Overlay_normal[normal_contours == 1] = np.array([0, 0, 0])
        Overlay_dyn_ws[dyn_ws_contours == 1] = np.array([0, 0, 0])

        out_patient = os.path.join(
            options.output, f.split('/')[-1]).replace('.png', '')

        CheckOrCreate(out_patient)

        def filename(name):
            return os.path.join(out_patient, name + ".png")
	bin[bin>0] = 255
        imsave(filename('rgb'), image)
        imsave(filename('binary'), bin)
        imsave(filename('prob'), prob)
        imsave(filename('OverlayNormal'), Overlay_normal)
        imsave(filename('OverlayDynWs'), Overlay_dyn_ws)
