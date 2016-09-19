from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import pdb


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(net, number_of_test, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in range(number_of_test):
        net.forward()
        # pdb.set_trace()
        hist += fast_hist(net.blobs[gt].data[0].flatten(),  # this was changed from .data[0,0].flatten()
                          net.blobs[layer].data[0].argmax(0).flatten(),
                          n_cl)

        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / number_of_test


def seg_tests(solver, number_of_test, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    hist, loss, acc, acc1, iu, fwavacc, recall, precision = do_seg_tests(solver.test_nets[0], solver.iter,
                                                                         number_of_test, layer, gt)
#    if validation_set is not None:
#        DataManagerForVal = SetupDataManager(options.path)
#        data_batch_size = 1
#        net = Net(model, options.weight, data_batch_size)
#        transformer = Transformer(net)
#        hist = compute_hist_VAL(net, DataManagerForVal,
#                                layer=options.scorelayer)
#        acc_, acc1_, iu_, fwavacc_, recall_, precision_ = Metrics(hist)
    return loss, acc, acc1, iu, fwavacc, recall, precision


def do_seg_tests(net, iter, number_of_test, layer='score', gt='label', verbose=True):

    n_cl = net.blobs[layer].channels

    hist, loss = compute_hist(net, number_of_test, layer, gt)
    acc = np.diag(hist).sum() / hist.sum()
    acc1 = np.diag(hist) / hist.sum(1)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    recall = (hist[1, 1] + 0.0) / (hist[1, 0] + hist[1, 1])
    precision = (hist[1, 1] + 0.0) / (hist[1, 1] + hist[0, 1])

    if verbose:
        print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
        print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
        print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc1)
        print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
        print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', fwavacc
        print '>>>', 'recall', recall
        print '>>>', 'precision', precision

    return hist, loss, acc, np.nanmean(acc1), np.nanmean(iu), fwavacc, recall, precision


def score_print(label_data, pred_bin):
    n_cl = len(np.unique(label_data))
    hist = fast_hist(label_data, pred_bin, n_cl)
    acc = (np.diag(hist).sum() + 0.0) / hist.sum()
    print '>>>', 'overall accuracy', acc
    acc1 = (np.diag(hist) + 0.0) / hist.sum(1)
    print '>>>', 'mean accuracy', np.nanmean(acc1)
    iu = np.diag(hist) / (0.0 + hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', 'mean IU', np.nanmean(iu)
    freq = (hist.sum(1) + 0.0) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print '>>>', 'fwavacc', fwavacc
    recall = (hist[1, 1] + 0.0) / (hist[1, 0] + hist[1, 1])
    precision = (hist[1, 1] + 0.0) / (hist[1, 1] + hist[0, 1])
    print '>>>', 'recall', recall
    print '>>>', 'precision', precision
