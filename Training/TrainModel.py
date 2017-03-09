import time
import os
import cPickle as pkl
from Solver import solver, run_solvers_IU
import numpy as np
import caffe
import pdb

from UsefulFunctions.RandomUtils import CheckOrCreate, CheckFile
from UsefulFunctions.EmailSys import ElaborateEmail

def TrainModel(options):
    wd = options.wd
    cn = options.cn
    dgtrain = options.dgtrain
    dgtest = options.dgtest
    patients = options.patients
    path_ = os.path.join(options.wd, options.cn)
    niter = options.niter
    disp_interval = options.disp_interval

    if options.hardware == "cpu":
        caffe.set_mode_cpu()
    elif options.hardware == "gpu":
        caffe.set_mode_gpu()

    for num in patients:
        try:
            datagen_test = pkl.load(open(dgtest, "rb"))
            datagen_train = pkl.load(open(dgtrain, "rb"))
            datagen_train.SetPatient(num)
            datagen_test.SetPatient(num)
            number_of_test = datagen_test.length
            pkl.dump(datagen_train, open(dgtrain, "w"))
            pkl.dump(datagen_test, open(dgtest, "w"))

            if options.epoch is not None:
                niter = options.epoch * number_of_test / options.batch_size

            if options.net != "FCN":
                weight = options.weight
                solver_path = os.path.join(path_, "solver.prototxt")
                train(solver_path, weight, wd, cn, niter,
                      disp_interval, number_of_test, num)
            else:
                before = None
                for i in options.archi:
                    fcn_num = "FCN{}".format(i)
                    if before is None:
                        weight = options.weight
                    else:
                        weight = os.path.join(
                            options.wd, options.cn, before, "temp_files", 'weights.{}_{}.caffemodel'.format(before, num))
                    path_ = os.path.join(wd, cn, fcn_num)
                    CheckOrCreate(path_)
                    solver_path = os.path.join(path_, "solver.prototxt")
                    cn_fcn = os.path.join(cn, fcn_num)
                    train(solver_path, weight, wd, cn_fcn, niter,
                          disp_interval, number_of_test, num)
                    before = fcn_num
        except Exception, e:
	    print e
            config = "PATIENT: {} \n \n "+ str(options) + " \n \n" + str(e)

            ElaborateEmail(config, "Error patient {}".format(num))


def train(solver_path, weight, wd, cn, niter, disp_interval, number_of_test, num):

    res_fold = os.path.join(wd, cn, "temp_files")

    ## Checking if file exists before running.
    name = cn
    if "/" in name:
        name = name.split('/')[-1]
    filename = 'weights.{}_{}.caffemodel'.format(name, num)
    if os.path.isfile(os.path.join(res_fold, filename)):
        print "file already exists"

    else:
        my_solver = caffe.get_solver(solver_path)

        if weight is not None:
            assert os.path.exists(weight)
            my_solver.net.copy_from(weight)

        start_time = time.time()
        print 'Running solvers for %d iterations...' % niter

        solvers = [(cn, my_solver)]


            
        Results, Results_train, weights = run_solvers_IU(
            niter, solvers, res_fold, disp_interval, number_of_test, num)

        Results.to_csv(os.path.join(
            res_fold, 'Metrics_{}_{}_{}.csv').format(disp_interval, niter, num))
        Results_train.to_csv(os.path.join(
            res_fold, 'MetricsTrain_{}_{}_{}.csv').format(disp_interval, niter, num))

        print 'Done.'

        diff_time = time.time() - start_time

        print ' \n '
        print 'Time for slide:'
        print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

        print ' \n '
        print "Average time per image: (have to put number of images.. "
        diff_time = diff_time / 10
        print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
