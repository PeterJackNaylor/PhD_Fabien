import caffe

import matplotlib.pylab as plt
import time
import os
import cPickle as pkl
from solver import solver, run_solvers, run_solvers_IU
import numpy as np
import pdb

def trainNet(kwargs):
    """
    options: cn : classifier name
             wd : work directory
             niter : number of iterations
             solver_path :  where to find the solver path
             weight : where to find the pre trained weights. (note: it can be "None")
             number_of_test: number of images in one epoch

    """
    cn = kwargs['cn']
    wd = kwargs['wd']
    gpu = kwargs['gpu']
    n_iter = kwargs['niter']
    solver_path = kwargs['solver_path']
    weight = kwargs['weight']
    disp_interval = kwargs['disp_interval']
    path_ = os.path.join(wd, cn)
    path_modelgen = os.path.join(path_, "model")
    datagen_path = os.path.join(path_modelgen, "data_generator_test.pkl")

    datagen_test = pkl.load(open(datagen_path, "rb"))
    number_of_test = datagen_test.length

    caffe.set_device(0)
    if gpu == "gpu":
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    if 'archi' not in kwargs.keys():
        train(solver_path, weight, wd, cn, n_iter,
              disp_interval, number_of_test)
    elif len(kwargs['archi']) == 1:
        train(solver_path, weight, wd, cn, n_iter,
              disp_interval, number_of_test)
    else:
        kwargs['archi'].sort()
        kwargs['archi'] = kwargs['archi'][::-1]
        first = True
        for num in kwargs['archi']:
            fcn_num = "FCN{}".format(num)
            solver_path = os.path.join(
                wd, cn, fcn_num, "solver.prototxt")
            if first:
                weight = weight
                before = fcn_num
                first = False
            else:
                weight = os.path.join(
                    wd, cn, before, "temp_files", "weights." + before + ".caffemodel")
                before = fcn_num

            train(solver_path, weight, wd, cn + '/' + fcn_num, n_iter,
                  disp_interval, number_of_test)
	    pdb.set_trace()

def train(solver_path, weight, wd, cn, niter, disp_interval, number_of_test):
    my_solver = caffe.get_solver(solver_path)
    # pdb.set_trace()

    if weight != "None":
        assert os.path.exists(weight)
        my_solver.net.copy_from(weight)

    start_time = time.time()
    print 'Running solvers for %d iterations...' % niter

    solvers = [(cn, my_solver)]

    res_fold = os.path.join(wd, cn, "temp_files")

    Results, Results_train, weights = run_solvers_IU(
        niter, solvers, res_fold, disp_interval, number_of_test)

    Results.to_csv(os.path.join(
        res_fold, 'Metrics_{}_{}.csv').format(disp_interval, niter))
    Results_train.to_csv(os.path.join(
        res_fold, 'MetricsTrain_{}_{}.csv').format(disp_interval, niter))

    print 'Done.'

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image: (have to put number of images.. "
    diff_time = diff_time / 10
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
