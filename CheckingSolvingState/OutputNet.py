import matplotlib as mpl

mpl.use('Agg')

import os
import glob
from skimage.io import imread as ir
from optparse import OptionParser
import time
from usefulPloting import Contours, ImageSegmentationSave
import glob
import caffe
import numpy as np


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)


def Transformer(net):
    mu = np.array([104.00698793, 116.66876762, 122.67891434])

# create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    # move image channels to outermost dimension
    transformer.set_transpose('data', (2, 0, 1))
    # subtract the dataset-mean value in each channel
    transformer.set_mean('data', mu)
    # rescale from [0, 1] to [0, 255]
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    return transformer


def LoadDataIntoNet(net, data, transformer):
    transformed_image_array = data.copy().swapaxes(1, 3)
    for i in range(transformed_image_array.shape[0]):
        transformed_image_array[i] = transformer.preprocess('data', data[i])
    net.blobs['data'].data[...] = transformed_image_array


def GetScoreVectors(net, data, transformer):

    LoadDataIntoNet(net, data, transformer)
    output = net.forward()
    score = output['score']

    return score


def LoadData(input_fold, range_min=0, range_max=None):
    files = glob.glob(os.path.join(input_fold, "*.png"))[range_min:range_max]
    data_batch_size = len(files)

    dimension = caffe.io.load_image(files[0]).shape

    out = np.zeros(shape=(data_batch_size, dimension[
                   0], dimension[1], dimension[2]))

    for i in range(len(files)):
        out[i, :, :, :] = caffe.io.load_image(files[i])
    return(files, out)


def all_in_one(net, input_fold, ouput, range_min=0, range_max=None):

    files, data = LoadData(input_fold, range_min, range_max)
    data_batch_size = len(files)
    net.blobs['data'].reshape(data_batch_size, 3, 512, 512)

    transformer = Transformer(net)
    SCOREZ = GetScoreVectors(net, data, transformer)

    for i in range(SCOREZ.shape[0]):
        score = SCOREZ[i, :, :, :]
        classed = np.argmax(score, axis=0)

        names = dict()
        all_labels = ["0: Background"] + ["1: Cell"]
        scores = np.unique(classed)
        labels = [all_labels[s] for s in scores]
        num_scores = len(scores)

        def rescore(c):
            """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
            return np.where(scores == c)[0][0]
        rescore = np.vectorize(rescore)

        painted = rescore(classed)

        ImageSegmentationSave(data[i, :, :, :], painted.transpose(),
                              os.path.join(ouput, files[i].split('/')[-1]))


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-o", "--output", dest="output",
                      help="Output folder")
    parser.add_option("-i", "--input", dest="input",
                      help="Input folder")
    parser.add_option('--weights', dest="weight",
                      help="Where the weights are.")
    parser.add_option('--model', dest='model',
                      help="Where the model is")
    parser.add_option('--batch_size', dest="batch",
                      help="batch size")
    (options, args) = parser.parse_args()

    # checking the input data

    CheckOrCreate(options.output)

    CheckExistants(options.input)
    CheckFile(options.weight)
    CheckFile(options.model)

    print "Input paramters to OutputNet:"
    print " \n "
    print "Input file        : | " + options.input
    print "Output folder     : | " + options.output
    print "Weight file       : | " + options.weight
    print "Model file        : | " + options.model
    print "Batch size        : | " + options.batch
    print ' \n '
    print "Beginning analyse:"

    start_time = time.time()
# Core of the code
########################################

    net = caffe.Net(options.model,      # defines the structure of the model
                    options.weight,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    files = glob.glob(os.path.join(options.input, "*.png"))
    n = len(files)
    i_old = 0
    for i in range(int(options.batch), n, int(options.batch)):
        all_in_one(net, options.input, options.output,
                   range_min=i_old, range_max=i)
        i_old = i
        ########################################

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for all:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image:"
    diff_time = diff_time / (n)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
