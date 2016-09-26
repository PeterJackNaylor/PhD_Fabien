import os
import numpy as np
from optparse import OptionParser

import pdb

create_dataset = True
create_net = True
create_solver = True
train = True


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option('--net', dest="net",
                      help="net architecture possible: FCN, DeconvNet, UNet")

    parser.add_option("--rawdata", dest="rawdata",
                      default='/data/users/pnaylor/Bureau/ToAnnotate',
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

    parser.add_option('--val_num', dest="val_num", default="1",
                      help="Number of images in test (times crop).")

    parser.add_option('--crop', dest="crop", default="1",
                      help="Number of crops by image, divided equally")

    parser.add_option('--solverrate', dest="solverrate", default="0.000001",
                      help="Initial rate of the solver at 10e-6")

    parser.add_option('--batch_size', dest="batch_size", default="1",
                      help="Size of the batches")

    parser.add_option('--size_x', dest="size_x", default=None)

    parser.add_option('--size_y', dest="size_y", default=None)

    parser.add_option('--img_format', dest="img_format", default="RGB",
                      help="Display image in RGB, HE or HEDab")

    parser.add_option('--loss', dest="loss", default="softmax",
                      help="loss possible: softmax or weight, or weight1")

    (options, args) = parser.parse_args()

    if options.wd is None:
        options.wd = '/home/naylor/Documents/Python/PhD/dataFCN'

    if options.niter is None:
        options.niter = "200"

    if options.cn is None:
        options.cn = 'NewClass'

    if options.solverrate != "0.000001":
        solverrate = 0.000001 * float(options.solverrate)
        options.solverrate = str(solverrate)

    print "Input paramters to run:"
    print " \n "
    print "Net used          : | " + options.net
    print "Raw data direct   : | " + options.rawdata
    print "Work directory    : | " + options.wd
    print "Classifier name   : | " + options.cn
    print "Weight file (init): | " + str(options.weight)
    print "Number of iteration | " + options.niter
    print "display interval  : | " + options.disp_interval
    print "Patients in test  : | " + options.val_num
    print "Number of crops   : | " + options.crop
    print "Solver rate       : | " + options.solverrate
    print "Sizes of batches  : | " + options.batch_size
    print "Image format      ; | " + options.img_format
    print "loss layer        : | " + options.loss

    if create_dataset:

        from WriteDataGen import WriteDataGen

        arg_datagen = {'wd': options.wd,
                       'cn': options.cn,
                       'rawdata': options.rawdata,
                       'val_num': options.val_num,
                       'seed': 42}

        if options.loss != "softmax":
            arg_datagen['loss'] = options.loss
        if options.crop != "1":
            arg_datagen['crop'] = int(options.crop)
        if options.size_x is not None:
            arg_datagen['crop_size'] = (
                int(options.size_x), int(options.size_y))
        if options.img_format != "RGB":
            arg_datagen['img_format'] = options.img_format

        WriteDataGen(arg_datagen)

    if create_net:

        arg_net = {'wd': options.wd,
                   'cn': options.cn,
                   'seed': 42}

        if options.loss != "softmax":
            arg_net['loss'] = options.loss

        if options.net == "UNet":

            from WriteUnet import WriteUnet

            WriteUnet(arg_net)

        if options.net == "DeconvNet":
            from WriteDeconvNet import WriteDeconvNet

            arg_net['batch_size'] = int(options.batch_size)

            WriteDeconvNet(arg_net)

    if create_solver:
        from solver import solver, run_solvers, run_solvers_IU
        solver_path = os.path.join(options.wd, options.cn, "solver.prototxt")
        outsnap = os.path.join(options.wd, options.cn, "snapshot")
        solverrate = float(options.solverrate)
        CheckOrCreate(os.path.join(options.wd, options.cn))
        CheckOrCreate(outsnap)

        name_solver = solver(solver_path,
                             os.path.join(options.wd, options.cn,
                                          "train.prototxt"),
                             test_net_path=os.path.join(
                                 options.wd, options.cn, "train.prototxt"),
                             base_lr=solverrate,
                             out_snap=outsnap)

    if train:
        from trainNet import trainNet
        arg_train = {}
        cn = options.cn
        wd = options.wd
        niter = int(options.niter)
        solver_path = os.path.join(options.wd, options.cn, "solver.prototxt")
        weight = options.weight
        disp_interval = int(options.disp_interval)
        trainNet(cn, wd, niter, solver_path, weight, disp_interval)
