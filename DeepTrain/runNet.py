import os
import numpy as np
from optparse import OptionParser
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
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

    parser.add_option("--epoch", dest="epoch", default="None",
                      help="Number of epoch, if epoch is specified, niter and disp_interval is dismissed")

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

    parser.add_option('--gpu', dest="gpu", default="gpu",
                      help="gpu or cpu")

    parser.add_option('--archi', dest="archi", default="32_16_8",
                      help=" you can specify, this parameter is only taken into account for \
                      FCN and valid : 32_16_8")

    parser.add_option('--momentum', dest="momentum", default="0.9",
                      help="momentum for the training")
    parser.add_option('--weight_decay', dest="weight_decay", default="0.0005",
                      help="weight decay for the training")
    parser.add_option('--stepsize', dest="stepsize", default="10000",
                      help="stepsize for the training")
    parser.add_option('--gamma', dest="gamma", default="0.1",
                      help="gamma for the training")
    parser.add_option('--enlarge', dest="enlarge", default="None",
                      help="enlarge with mirrored image")
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

    if options.epoch != "None":
        options.niter = options.epoch + " epoch"
        options.epoch = int(options.epoch)
        options.disp_interval = "1 epoch"

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
    print "gpu or cpu        : | " + options.gpu
    print "Momentul          : | " + options.momentum
    print "weight decay      : | " + options.weight_decay
    print "stepsize          : | " + options.stepsize
    print "gamma             : | " + options.gamma
    print "epoch             : | " + str(options.epoch)
    print "enlarge           : | " + options.enlarge

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
        if options.net == "UNet":
            arg_datagen['UNet'] = True
        else:
            arg_datagen['UNet'] = False
        if options.enlarge == "True":
            arg_datagen["enlarge"] = True
        WriteDataGen(arg_datagen)

    if create_net:

        arg_net = {'wd': options.wd,
                   'cn': options.cn,
                   'seed': 42,
                   'batch_size': int(options.batch_size)}

        if options.loss != "softmax":
            arg_net['loss'] = options.loss

        if options.net == "UNet":

            from WriteUnet import WriteUnet

            WriteUnet(arg_net)

        if options.net == "DeconvNet":

            from WriteDeconvNet import WriteDeconvNet

            WriteDeconvNet(arg_net)

        if options.net == "FCN":

            from WriteFCN import WriteFCN

            archi = [int(el) for el in options.archi.split('_')]
            arg_net['archi'] = archi

            WriteFCN(arg_net)

    if create_solver:
        from WriteSolver import WriteSolver

        arg_solver = arg_net
        arg_solver["solverrate"] = float(options.solverrate)
        arg_solver["momentum"] = float(options.momentum)
        arg_solver["weight_decay"] = float(options.weight_decay)
        arg_solver["stepsize"] = int(options.stepsize)
        arg_solver["momentum"] = float(options.momentum)
        arg_solver["gamma"] = float(options.gamma)
        if options.net == "FCN":
            archi = [int(el) for el in options.archi.split('_')]
            if len(archi) != 1:
                arg_solver['archi'] = [int(el)
                                       for el in options.archi.split('_')]

        WriteSolver(arg_solver)

    if train:
        from trainNet import trainNet
        arg_train = arg_net
        if options.epoch != "None":
            arg_train["epoch"] = int(options.epoch)
            arg_net['batch_size'] = int(options.batch_size)
        else:
            arg_train['niter'] = int(options.niter)
            arg_train['disp_interval'] = int(options.disp_interval)

        arg_train['solver_path'] = os.path.join(
            options.wd, options.cn, "solver.prototxt")
        arg_train['weight'] = options.weight
        arg_train['gpu'] = options.gpu
        if options.net == "FCN":
            archi = [int(el) for el in options.archi.split('_')]
            if len(archi) != 1:
                arg_train['archi'] = [int(el)
                                      for el in options.archi.split('_')]

        trainNet(arg_train)
