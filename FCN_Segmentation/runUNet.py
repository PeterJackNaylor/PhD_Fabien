# -*- coding: utf-8 -*-

"""

line for kepler:

export WEIGHT=/data/users/pnaylor/Documents/Python/FCN/model/fcn32s-heavy-pascal.caffemodel
export WD=/data/users/pnaylor/Documents/Python/FCN
export RAWDATA=/data/users/pnaylor/Bureau/ToAnnotate

--rawdata $RAWDATA --wd $WD --cn FCN1 --weight $WEIGHT --niter 200

"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import Unet

# from DataToLMDB import MakeDataLikeFCN
import ImageTransf as Transf
import caffe
import os
import cPickle as pkl
from DataLayerPeter import DataGen
from solver import solver, run_solvers, run_solvers_IU

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
                      help="Classifier name, like FCN32")

    parser.add_option("--weight", dest="weight", default="None",
                      help="Where to find the weight file")

    parser.add_option("--niter", dest="niter",
                      help="Number of iterations")

    parser.add_option("--disp_interval", dest="disp_interval",
                      help=" Diplay interval for training the network", default="10")

    parser.add_option("--layer_score", dest="scorelayer",
                      help="name of score layer", default="score")

    parser.add_option('--val_test', dest="val_num", default="6",
                      help="Number of images in test (times crop).")

    parser.add_option('--crop', dest="crop", default="1",
                      help="Number of crops by image, divided equally")

    parser.add_option('--solverrate', dest="solverrate", default="0.000001",
                      help="Initial rate of the solver at 10e-6")

    parser.add_option('--template', dest="template", default="None",
                      help="Folder where the tempaltes are")

    parser.add_option('--batch_size', dest="batch_size", default="1",
                      help="Size of the batches")
    parser.add_option('--size_x', dest="size_x", default=None)
    parser.add_option('--size_y', dest="size_y", default=None)

    parser.add_option('--img_format', dest="img_format", default="RGB",
                      help="Display image in RGB, HE or HEDab")
    parser.add_option('--loss', dest="loss", default="softmax",
                      help="loss possible: softmax or weight")
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
        options.cn = 'NewClass'

    if options.solverrate != "0.000001":
        solverrate = 0.000001 * float(options.solverrate)
        options.solverrate = str(solverrate)

    if options.size_x is None:
        crop_size = None
        print_crop = "No random cropping"
    else:
        crop_size = (int(options.size_x), int(options.size_y))
        print_crop = options.size_x + " x " + options.size_y

    print "Input paramters to run:"
    print " \n "
    print "Raw data direct   : | " + options.rawdata
    print "Work directory    : | " + options.wd
    print "Classifier name   : | " + options.cn
    print "Weight file (init): | " + options.weight
    print "Number of iteration | " + options.niter
    print "display interval  : | " + options.disp_interval
    print "score layer name  : | " + options.scorelayer
    print "Patients in test  : | " + options.val_num
    print "Number of crops   : | " + options.crop
    print "Solver rate       : | " + options.solverrate
    print "path to template  : | " + options.template
    print "Sizes of batches  : | " + options.batch_size
    print "random crop size  : | " + print_crop
    print "Image format      ; | " + options.img_format
    print "loss layer        : | " + options.loss
    if options.crop == "1":
        crop = None
    else:
        crop = int(options.crop)

    options.niter = int(options.niter)

    create_dataset = True
    create_solver = True
    create_net = True  # False
    create_net_cheat = False

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
        if options.loss == "weight":
            Weight = True
        else:
            Weight = False
        data_generator_train = DataGen(options.rawdata, crop=crop, size=crop_size,
                                       transforms=transform_list, split="train",
                                       leave_out=int(options.val_num), seed=42,
                                       img_format=options.img_format, Weight=Weight)
        pkl.dump(
            data_generator_train, open(os.path.join(path_modelgen, "data_generator_train.pkl"), "wb"))
        data_generator_test = DataGen(options.rawdata, crop=crop, size=crop_size,
                                      transforms=[Transf.Identity()], split="test",
                                      leave_out=int(options.val_num), seed=42,
                                      img_format=options.img_format, Weight=Weight)
        pkl.dump(
            data_generator_test, open(os.path.join(path_modelgen, "data_generator_test.pkl"), "wb"))
    if create_net:
        path_modelgen = os.path.join(options.wd, options.cn, "model")
        # os.path.join(options.wd, "train")

        datagen_path = os.path.join(path_modelgen, "data_generator_train.pkl")
        CheckOrCreate(os.path.join(options.wd, options.cn))
        Unet.make_net(os.path.join(options.wd, options.cn),
                      datagen_path,
                      os.path.join(
            path_modelgen, "data_generator_test.pkl"),
            classifier_name=options.cn
            loss_layer=options.loss)

    solver_path = os.path.join(options.wd, options.cn, "solver.prototxt")
    outsnap = os.path.join(options.wd, options.cn, "snapshot", "snapshot")

    CheckOrCreate(os.path.join(options.wd, options.cn))
    CheckOrCreate(outsnap)

    if create_solver:
        name_solver = solver(solver_path,
                             os.path.join(options.wd, options.cn,
                                          "train.prototxt"),
                             test_net_path=os.path.join(
                                 options.wd, options.cn, "train.prototxt"),
                             # batch_size=int(options.batch_size),
                             base_lr=solverrate,
                             out_snap=outsnap)
        # name_solver is solver_path.....

    caffe.set_device(0)
    caffe.set_mode_gpu()

    niter = options.niter

    pdb.set_trace()
    my_solver = caffe.get_solver(solver_path)
    # pdb.set_trace()

    if options.weight != "None":
        weights = options.weight
        assert os.path.exists(weights)
        my_solver.net.copy_from(weights)

    start_time = time.time()
    print 'Running solvers for %d iterations...' % niter

    solvers = [(options.cn, my_solver)]

    res_fold = os.path.join(options.wd, options.cn, "temp_files")
    number_of_test = data_generator_test.length
    loss, acc, acc1, iu, fwavacc, recall, precision, weights = run_solvers_IU(
        niter, solvers, res_fold, int(options.disp_interval), number_of_test, options.scorelayer)

    np.save(os.path.join(res_fold, "loss"), loss[options.cn])

    np.save(os.path.join(res_fold, "acc"), acc[options.cn])
    np.save(os.path.join(res_fold, "acc1"), acc1[options.cn])
    np.save(os.path.join(res_fold, "iu"), iu[options.cn])
    np.save(os.path.join(res_fold, "fwavacc"), fwavacc[options.cn])
    np.save(os.path.join(res_fold, "precision"), precision[options.cn])
    np.save(os.path.join(res_fold, "recall"), recall[options.cn])

    range_iter = range(int(options.disp_interval), niter + int(options.disp_interval),
                       int(options.disp_interval))
    pref = options.cn
    plt.plot(range_iter, loss[pref])
    plt.savefig(os.path.join(res_fold, "loss"))
    plt.close()
    plt.plot(range_iter, acc[pref], "-r")
    plt.plot(range_iter, acc1[pref], "-y")
    plt.plot(range_iter, iu[pref], "-g")
    plt.plot(range_iter, fwavacc[pref], "-b")
    plt.plot(range_iter, recall[pref], "-c")
    plt.plot(range_iter, precision[pref], "-m")
    plt.savefig(os.path.join(res_fold, "alltogethernow"))
    print 'Done.'

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image: (have to put number of images.. "
    diff_time = diff_time / 10
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)