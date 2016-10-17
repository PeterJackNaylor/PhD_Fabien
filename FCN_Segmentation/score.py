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
#        pdb.set_trace()
        hist += fast_hist(net.blobs[gt].data[0].flatten(),  # this was changed from .data[0,0].flatten()
                          net.blobs[layer].data[0].argmax(0).flatten(),
                          n_cl)
#	if np.isnan(net.blobs['loss'].data.flat[0]):
#	    pdb.set_trace()
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / number_of_test


def seg_tests(solver, number_of_test, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    hist, metrics = do_seg_tests(solver.test_nets[0], solver.iter,
                                 number_of_test, layer, gt, "Test Net:")
    hist_train, metrics_train = do_seg_tests(solver.net, solver.iter,
                                             number_of_test, layer, gt, "Train Net:")
#    if validation_set is not None:
#        DataManagerForVal = SetupDataManager(options.path)
#        data_batch_size = 1
#        net = Net(model, options.weight, data_batch_size)
#        transformer = Transformer(net)
#        hist = compute_hist_VAL(net, DataManagerForVal,
#                                layer=options.scorelayer)
#        acc_, acc1_, iu_, fwavacc_, recall_, precision_ = Metrics(hist)
    return metrics, metrics_train


def do_seg_tests(net, iter, number_of_test, layer='score', gt='label', id="test", verbose=True):

    n_cl = net.blobs[layer].channels

    hist, loss = compute_hist(net, number_of_test, layer, gt)
    metrics = []
    acc = np.diag(hist).sum() / hist.sum()
    metrics.append((acc, 'acc'))
    metrics.append((np.diag(hist) / hist.sum(1), "acc1"))
    iu = np.diag(hist) / (hist.sum(1) +
                          hist.sum(0) - np.diag(hist))
    metrics.append((iu, "iu"))
    freq = hist.sum(1) / hist.sum()
    metrics.append((freq, "freq"))
    metrics.append(((freq[freq > 0] * iu[freq > 0]).sum(), "fwavacc"))
    recall = (hist[1, 1] + 0.0) / (hist[1, 0] + hist[1, 1])
    metrics.append((recall, "recall"))
    metrics.append(
        ((hist[1, 1] + 0.0) / (hist[1, 1] + hist[0, 1]), "precision"))
    metrics.append((recall, 'true positive'))
    true_neg = (hist[0, 0] + 0.0) / (hist[0, 1] + hist[0, 0])
    metrics.append((true_neg, 'true negatives'))
    metrics.append(((recall + true_neg) / 2, 'performance'))
    metrics.append((1 - acc, 'pixel error'))
    if verbose:
        print ">>>", datetime.now(), "Iteration", iter, "for", id
        for val, name in metrics:
            print '>>>', name, val
    return hist, metrics


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
