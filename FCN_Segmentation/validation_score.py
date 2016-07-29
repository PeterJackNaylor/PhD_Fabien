import caffe
import numpy as np
from ShortPrediction import Net
from CheckingSolvingState.OutputNet import Transformer, GetScoreVectors
from Datamanager import DataManager
from optparse import OptionParser
import time
from score import fast_hist


def Pred(net, img, transformer):
        # image =
        # caffe.io.load_image('../data/pascal-voc2010/JPEGImages/2007_000241.jpg')
    transformed_image = transformer.preprocess('data', img)

    net.blobs['data'].data[...] = transformed_image
    output = net.forward()

    score = output['score'][0]
    classed = np.argmax(score, axis=0)

    all_labels = ["0: Background"] + ["1: Cell"]
    scores = np.unique(classed)
    labels = [all_labels[s] for s in scores]
    num_scores = len(scores)

    def rescore(c):
        """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
        return np.where(scores == c)[0][0]

    rescore = np.vectorize(rescore)

    painted = rescore(classed)

    return painted


def ComputeHist(img_pred, img_gt, hist, n_cl):
    hist += fast_hist(img_gt.flatten(),
                      img_pred.flatten(),
                      n_cl)
    return hist


def compute_hist_VAL(net, dataM, layer="score"):
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))

    for img, img_gt, name in dataM.TrainingIteratorLeaveValOut():
        img_pred = Pred(net, img, transformer)
        hist += ComputeHist(img_pred, img_gt, hist, n_cl)

    return hist


def SetupDataManager(path):
    datatest = DataManager(path)
    datatest.prepare_sets(leave_out=0)
    datatest.SetTransformation(None)
    return datatest


def Metrics(hist):
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', 'overall accuracy', acc
    # per-class accuracy
    pdb.set_trace()
    acc1 = np.diag(hist) / hist.sum(1)
    print '>>>', 'mean accuracy', np.nanmean(acc1)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print '>>>', 'fwavacc', fwavacc

    return acc, np.nanmean(acc1), np.nanmean(iu), fwavacc


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-p", "--path", dest="path",
                      help="Input path (raw data)")
    parser.add_option('-w', '--width', dest="width", default="512",
                      help="width")
    parser.add_option('--heigth', dest="heigth", default="512",
                      help="heigth")
    parser.add_option("--weight", dest="weight",
                      help="Where to find the weight file")
    parser.add_option("--layer_score", dest="scorelayer",
                      help="name of score layer", default="score")
    parser.add_option("--model_def", dest="model_def",
                      help="Model definition, prototxt file")

    (options, args) = parser.parse_args()

    print " \n "
    print "Input paramters to Validation score:"
    print " \n "
    print "Input file        : | " + options.path
    print "Heigth            : | " + options.heigth
    print "Width             : | " + options.width
    print "score layer name  : | " + options.scorelayer
    print "Weight file (init): | " + options.weight
    print "Model definition  : | " + options.model_def
    start_time = time.time()

    DataManagerForVal = SetupDataManager(options.path)
    data_batch_size = 1
    net = Net(options.model_def, options.weight, data_batch_size)
    transformer = Transformer(net)
    hist = compute_hist_VAL(net, DataManagerForVal, layer=options.scorelayer)
    acc, acc1, iu, fwavacc = Metrics(hist)

    print 'Done.'

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
