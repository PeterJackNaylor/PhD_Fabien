# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:07:56 2016

@author: naylor
"""
import os
from caffe.proto import caffe_pb2
import pdb
import numpy as np

import score


def solver(solver_name, train_net_path, test_net_path=None, base_lr=0.001, out_snap="./temp_snapshot"):
    # pdb.set_trace()
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100)  # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    s.max_iter = 50000    # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    if not os.path.isdir(out_snap):
        os.mkdir(out_snap)
    s.snapshot_prefix = out_snap

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    with open(solver_name, 'w') as f:
        f.write(str(s))
        return f.name


def run_solvers_IU(niter, solvers, res_fold, disp_interval, val, layer):
    val = np.loadtxt(val, dtype=str)
    blobs = ('loss', 'acc', 'acc1', 'iu', 'fwavacc', 'recall', 'precision')
    number_of_loops = niter / disp_interval

    loss, acc, acc1, iu, fwavacc, recall, precision = ({name: np.zeros(number_of_loops) for name, _ in solvers}
                                                       for _ in blobs)
    for it in range(number_of_loops):
        for name, s in solvers:
            # pdb.set_trace()
            s.step(disp_interval)  # run a single SGD step in Caffe
            # DEFINE VAL is validation test set, it computes it independently
            # ...
            loss[name][it], acc[name][it], acc1[name][it], iu[name][it], fwavacc[
                name][it], recall[name][it], precision[
                name][it] = score.seg_tests(s, False, val, layer=layer)
    # Save the learned weights from all nets.
    if not os.path.isdir(res_fold):
        os.mkdir(res_fold)
    weight_dir = res_fold
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, acc1, iu, fwavacc, recall, precision, weights


def run_solvers(niter, solvers, res_fold, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            # pdb.set_trace()
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = [s.net.blobs[b].data.copy()
                                             for b in blobs]
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100 * acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from both nets.
    if not os.path.isdir(res_fold):
        os.mkdir(res_fold)
    weight_dir = res_fold
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights
