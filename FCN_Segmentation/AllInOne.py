import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt


import sys
sys.path[4] = '/data/users/pnaylor/Documents/Python/caffe_peter/python'

import cPickle as pkl
from DataLayerPeter import DataGen
import ImageTransf as Transf
import caffe
import os
from solver import solver, run_solvers, run_solvers_IU
import FCN32
import FCN16
import FCN8
import numpy as np
import time
from optparse import OptionParser

import pdb


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--rawdata", dest="rawdata",
                      help="raw data folder, with respect to datamanager.py")

    parser.add_option("--wd", dest="wd",
                      help="Working directory")

    parser.add_option("--cn", dest="cn",
                      help="Name of procedure")

    parser.add_option("--weight", dest="weight",
                      help="Where to find the weight file")

    parser.add_option("--niter", dest="niter",
                      help="Number of iterations")

    parser.add_option("--disp_interval", dest="disp_interval",
                      help=" Diplay interval for training the network", default="10")

    parser.add_option("--layer_score", dest="scorelayer",
                      help="name of score layer", default="score")

    parser.add_option('--gpu', dest="gpu",
                      help="Which GPU to use.")

    parser.add_option('--val_test', dest="val_num", default="6",
                      help="Number of images in test (times crop).")

    parser.add_option('--crop', dest="crop", default="1",
                      help="Number of crops by image, divided equally")

    parser.add_option('--solverrate', dest="solverrate", default="0.000001",
                      help="Initial rate of the solver at 10e-6")

    (options, args) = parser.parse_args()

    if options.rawdata is None:
        options.rawdata = '/home/naylor/Bureau/ToAnnotate'

    if options.wd is None:
        options.wd = '/home/naylor/Documents/Python/PhD/dataFCN'

    if options.weight is None:
        options.weight = '/home/naylor/Documents/Python/PhD/dataFCN/PretrainedWeights/fcn32s-heavy-pascal.caffemodel'

    if options.niter is None:
        options.niter = "200"

    if options.cn is None:
        options.cn = 'fcn32'

    if options.gpu is None:
        options.gpu = "0"

    if options.solverrate != "0.000001":
        solverrate = 0.000001 * float(options.solverrate)
        options.solverrate = str(solverrate)

    print "Input paramters to run:"
    print " \n "
    print "Raw data direct   : | " + options.rawdata
    print "Work directory    : | " + options.wd
    print "Classifier name   : | " + options.cn
    print "Weight file (init): | " + options.weight
    print "Number of iteration | " + options.niter
    print "Which GPU         : | " + options.gpu
    print "display interval  : | " + options.disp_interval
    print "score layer name  : | " + options.scorelayer
    print "Images in test    : | " + options.val_num
    print "Number of crops   : | " + options.crop
    print "Solver rate       : | " + options.solverrate

    if options.crop == "1":
        crop = None
    else:
        crop = int(options.crop)

    options.niter = int(options.niter)

    create_dataset = True
    create_solver = True
    create_net = True  # False

    enlarge = False  # create symetry if the image becomes black ?

    transform_list = [Transf.Identity(),
                      Transf.Flip(0),
                      Transf.Flip(1),
                      Transf.OutOfFocus(5),
                      Transf.OutOfFocus(10),
                      Transf.ElasticDeformation(0, 30, num_points=4),
                      Transf.ElasticDeformation(0, 30, num_points=4)]

    for rot in range(1, 360):
        transform_list.append(Transf.Rotation(rot, enlarge=enlarge))
    for sig in [1, 2, 3, 4]:
        transform_list.append(Transf.OutOfFocus(sig))
    for i in range(20):
        transform_list.append(Transf.ElasticDeformation(0, 30, num_points=4))

    if create_dataset:
        path_modelgen = os.path.join(options.wd, options.cn, "model")
        CheckOrCreate(path_modelgen)
        data_generator_train = DataGen(options.rawdata, crop=crop,
                                       transforms=transform_list, split="train", leave_out=int(options.val_num), seed=42)
        pkl.dump(
            data_generator_train, open(os.path.join(path_modelgen, "data_generator_train.pkl"), "wb"))
        data_generator_test = DataGen(options.rawdata, crop=crop,
                                      transforms=[Transf.Identity()], split="test", leave_out=int(options.val_num), seed=42)
        pkl.dump(
            data_generator_test, open(os.path.join(path_modelgen, "data_generator_test.pkl"), "wb"))

    if create_net:
        datagen_path = os.path.join(path_modelgen, "data_generator_train.pkl")
        datagen_path_test = os.path.join(
            path_modelgen, "data_generator_test.pkl")
        CheckOrCreate(os.path.join(options.wd, options.cn, "FCN32"))
        FCN32.make_net(os.path.join(options.wd, options.cn, "FCN32"),
                       datagen_path,
                       datagen_path_test,
                       classifier_name=options.cn,
                       classifier_name1="score_fr1",
                       classifier_name2="upscore1")
        CheckOrCreate(os.path.join(options.wd, options.cn, "FCN16"))
        FCN16.make_net(os.path.join(options.wd, options.cn, "FCN16"),
                       datagen_path,
                       datagen_path_test,
                       classifier_name=options.cn,
                       classifier_name1="score_fr1")
        CheckOrCreate(os.path.join(options.wd, options.cn, "FCN8"))
        FCN8.make_net(os.path.join(options.wd, options.cn, "FCN8"),
                      datagen_path,
                      datagen_path_test,
                      classifier_name=options.cn,
                      classifier_name1="score_fr1",
                      classifier_name2="upscore2",
                      classifier_name3="score_pool4")

    solver_path = os.path.join(options.wd, options.cn, "solver.prototxt")
    outsnap = os.path.join(options.wd, options.cn, "snapshot")

    CheckOrCreate(os.path.join(options.wd, options.cn))
    CheckOrCreate(outsnap)

    if create_solver:
        for pref in ["FCN32", "FCN16", "FCN8"]:
            solver_path = os.path.join(
                options.wd, options.cn, pref, "solver.prototxt")
            outsnap = os.path.join(
                options.wd, options.cn, pref, "snapshot", "snapshot")
            CheckOrCreate(os.path.join(options.wd, options.cn, pref))
            CheckOrCreate(outsnap)
            name_solver = solver(solver_path,
                                 os.path.join(options.wd, options.cn, pref,
                                              "train.prototxt"),
                                 # os.path.join(options.wd, options.cn, pref, "train.prototxt"),
                                 test_net_path=os.path.join(options.wd, options.cn, pref,
                                                            "test.prototxt"),
                                 base_lr=solverrate,
                                 out_snap=outsnap)
        # name_solver is solver_path.....
    weights = options.weight
    assert os.path.exists(weights)

    caffe.set_device(int(options.gpu))
    caffe.set_mode_gpu()

    niter = options.niter

    def res_fold(stride):
        return os.path.join(options.wd, options.cn, stride, "temp_files")
    # pdb.set_trace()
    w_d = {"FCN32": weights,
           "FCN16": os.path.join(res_fold("FCN32"), "weights.FCN32.caffemodel"),
           "FCN8":  os.path.join(res_fold("FCN16"), "weights.FCN16.caffemodel")}
    r_f = {"FCN32": res_fold("FCN32"),
           "FCN16": res_fold("FCN16"),
           "FCN8":  res_fold("FCN8")}

    def solv_path(stride):
        return os.path.join(options.wd, options.cn, stride, "solver.prototxt")
    s_d = {"FCN32": solv_path("FCN32"), "FCN16": solv_path(
        "FCN16"), "FCN8": solv_path("FCN8")}

    start_time = time.time()
    print 'Running solvers for %d iterations...' % niter

    range_iter = range(int(options.disp_interval), niter + int(options.disp_interval),
                       int(options.disp_interval))
    # pdb.set_trace()
    for pref in ["FCN32", "FCN16", "FCN8"]:
        my_solver = caffe.get_solver(s_d[pref])
        # pdb.set_trace()
        my_solver.net.copy_from(w_d[pref])

        solvers = [(pref, my_solver)]
        number_of_test = data_generator_test.length
        loss, acc, acc1, iu, fwavacc, recall, precision, weights = run_solvers_IU(
            niter, solvers, r_f[pref], int(options.disp_interval), number_of_test, options.scorelayer)

        np.save(os.path.join(r_f[pref], "loss"), loss[pref])

        np.save(os.path.join(r_f[pref], "acc"), acc[pref])
        np.save(os.path.join(r_f[pref], "acc1"), acc1[pref])
        np.save(os.path.join(r_f[pref], "iu"), iu[pref])
        np.save(os.path.join(r_f[pref], "fwavacc"), fwavacc[pref])
        np.save(os.path.join(r_f[pref], "precision"), precision[pref])
        np.save(os.path.join(r_f[pref], "recall"), recall[pref])

        plt.plot(range_iter, loss[pref])
        plt.savefig(os.path.join(r_f[pref], "loss"))
        plt.close()
        plt.plot(range_iter, acc[pref], "-r")
        plt.plot(range_iter, acc1[pref], "-y")
        plt.plot(range_iter, iu[pref], "-g")
        plt.plot(range_iter, fwavacc[pref], "-b")
        plt.plot(range_iter, recall[pref], "-c")
        plt.plot(range_iter, precision[pref], "-m")
        plt.savefig(os.path.join(r_f[pref], "alltogethernow"))
