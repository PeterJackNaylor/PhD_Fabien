from __future__ import division
import numpy as np
from datetime import datetime
import pdb

from RandomUtils import fast_hist

def compute_hist(net, number_of_test, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in range(number_of_test):
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0].flatten(),  # this was changed from .data[0,0].flatten()
                          net.blobs[layer].data[0].argmax(0).flatten(),
                          n_cl)

        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / number_of_test

def seg_tests(solver, number_of_test, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    hist, metrics = do_seg_tests(solver.test_nets[0], solver.iter,
                                 number_of_test, layer, gt, "Test Net:")
    hist_train, metrics_train = do_seg_tests(solver.net, solver.iter,
                                             number_of_test, layer, gt, "Train Net:")

    return metrics, metrics_train

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

def do_seg_tests(net, iter, number_of_test, layer='score', gt='label', id="test", verbose=True):
    n_cl = net.blobs[layer].channels
    hist, loss = compute_hist(net, number_of_test, layer, gt)

    metrics = []
    metrics.append((loss, 'Loss'))
    acc = np.diag(hist).sum() / hist.sum()
    metrics.append((acc, 'Overall accuracy:'))
    metrics.append((1 - acc, 'Pixel error'))
    acc1 = (np.diag(hist) + 0.0) / hist.sum(1)
    metrics.append((np.nanmean(acc1), "Mean accuracy:"))
    iu = np.diag(hist) / (hist.sum(1) +
                          hist.sum(0) - np.diag(hist))
    metrics.append((np.nanmean(iu), "Intersection Over Union : "))
    freq = hist.sum(1) / hist.sum()
    metrics.append(((freq[freq > 0] * iu[freq > 0]).sum(), "fwavacc"))

    if n_cl == 2 :
        recall = (hist[1, 1] + 0.0) / (hist[1, 0] + hist[1, 1])
        metrics.append((recall, "Recall"))
        prec = (hist[1, 1] + 0.0) / (hist[1, 1] + hist[0, 1])
        metrics.append((prec, "Precision"))
        metrics.append((recall, 'True positive'))
        true_neg = (hist[0, 0] + 0.0) / (hist[0, 1] + hist[0, 0])
        metrics.append((true_neg, 'True negatives'))
        metrics.append(((recall + true_neg) / 2, 'Performance'))
        F1 = 2 * prec * recall / (prec + recall)
        metrics.append((F1, 'F1'))
    if verbose:
        print "\n "
        print ">>>", datetime.now(), "Iteration", iter, "for", id
        for val, name in metrics:
            print '>>>', name, val
        if n_cl != 2:
            print ">>> Confusion matrix:"
            labels = ["Background", "Fat cells", "Cancerous", "Lymphocyte", "Fibroblast",
                      "Mitosis", "Epithelial", "Normal", "Ignore", "Necrose"]
            print_cm(hist, labels)
            #metrics.append((hist))
    return hist, metrics


def score_print(label_data, pred_bin):
    ### should i keep this function?
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
