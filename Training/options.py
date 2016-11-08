from optparse import OptionParser
from UsefulFunctions import ImageTransf as Transf
import os
import numpy as np


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def GetOptions(verbose=True):

    parser = OptionParser()

# complsory aguments:
    parser.add_option('--net', dest="net", type="string",
                      help="net architecture possible: FCN, DeconvNet, UNet, BaochuanNet")
    parser.add_option("--rawdata", dest="rawdata", type="string",
                      default='/data/users/pnaylor/Bureau/ToAnnotate',
                      help="raw data folder, with respect to datamanager.py")
    parser.add_option("--wd", dest="wd",
                      help="Working directory")
    parser.add_option("--cn", dest="cn",
                      help="Classifier name, like FCN32")

# options with default values

    parser.add_option('--leaveout', dest="leaveout", default=1, type="int",
                      help="Number of images in test (times crop).")
    parser.add_option("--niter", dest="niter", default=20000, type="int",
                      help="Number of iterations")
    parser.add_option("--disp_interval", dest="disp_interval", default=100, type="int",
                      help=" Diplay interval for training the network")
    parser.add_option('--crop', dest="crop", default=1, type="int",
                      help="Number of crops by image, divided equally")
    parser.add_option('--base_lr', dest="base_lr", default=1.0, type="float",
                      help="Initial rate of the solver at 10e-6")
    parser.add_option('--batch_size', dest="batch_size", default=1, type="int",
                      help="Size of the batches")
    parser.add_option('--img_format', dest="img_format", default="RGB", type="string",
                      help="Display image in RGB, HE or HEDab")
    parser.add_option('--loss', dest="loss", default="softmax", type="string",
                      help="loss possible: softmax or weight, or weight1")
    parser.add_option('--momentum', dest="momentum", default=0.9, type="float",
                      help="momentum for the training")
    parser.add_option('--weight_decay', dest="weight_decay", default=0.0005, type="float",
                      help="weight decay for the training")
    parser.add_option('--stepsize', dest="stepsize", default=10000, type="int",
                      help="stepsize for the training")
    parser.add_option('--gamma', dest="gamma", default=0.1, type="float",
                      help="gamma for the training")
    parser.add_option('--enlarge', dest="enlarge", default=1, type="int",
                      help="enlarge with mirrored image")
    parser.add_option('--seed', dest="seed", default=1337, type="int",
                      help="seed for datalayer")

# non compulsory arguments with no default
    parser.add_option("--weight", dest="weight", type="string",
                      help="Where to find the weight file")
    parser.add_option("--epoch", dest="epoch", type="int",
                      help="Number of epoch, if epoch is specified, niter and disp_interval is dismissed")
    parser.add_option('--size_x', dest="size_x", type="int")
    parser.add_option('--size_y', dest="size_y", type="int")

 # arguments that need to be reprocessed

    parser.add_option('--archi', dest="archi", default="32_16_8",
                      help=" you can specify, this parameter is only taken into account for \
                      FCN and valid : 32_16_8")
    parser.add_option('--skip', dest="skip", default="", type="string",
                      help=" Which layers to skip, give 1234 or 123, 13, 12, 1 etc .")

    (options, args) = parser.parse_args()

    print "Input paramters to run:"
    print " \n "

# complsory aguments:

    print "Net used          : | {}".format(options.net)
    print "Raw data direct   : | {}".format(options.rawdata)
    print "Work directory    : | {}".format(options.wd)
    CheckOrCreate(options.wd)
    CheckOrCreate(os.path.join(options.wd, options.cn))
    print "Classifier name   : | {}".format(options.cn)

# options with default values

    print "Patients in test  : | {}".format(options.leaveout)
    print "Number of iteration | {}".format(options.niter)
    print "display interval  : | {}".format(options.disp_interval)
    print "Number of crops   : | {}".format(options.crop)
    options.base_lr = 0.000001 * options.base_lr
    print "base_lr           : | {}".format(options.base_lr)
    print "Sizes of batches  : | {}".format(options.batch_size)
    print "Image format      ; | {}".format(options.img_format)
    print "loss layer        : | {}".format(options.loss)
    print "Momentul          : | {}".format(options.momentum)
    print "weight decay      : | {}".format(options.weight_decay)
    print "stepsize          : | {}".format(options.stepsize)
    print "gamma             : | {}".format(options.gamma)
    print "enlarge           : | {}".format(options.enlarge)
    print "seed              : | {}".format(options.seed)

# non compulsory arguments with no default

    if options.weight is not None:
        print "Weight file (init): | {}".format(options.weight)
    else:
        print "NO WEIGHT"
    if options.epoch is not None:
        print "epoch             : | {}".format(options.epoch)
    else:
        print "NO EPOCH"
    if options.size_x is not None or options.size_y is not None:
        print "size              : | ({},{})".format(options.size_x, options.size_y)
        options.crop_size = (options.size_x, options.size_y)
    else:
        options.crop_size = None
        print "NO RANDOM CROP"

 # arguments that need to be reprocessed

    if options.enlarge == 1:
        options.enlarge = True
    else:
        options.enlarge = False

    print "archi             : | {}".format(options.archi)
    options.archi = [int(el) for el in options.archi.split('_')]
    print "skip              : | {}".format(options.skip)
    options.skip = [int(el) for el in options.skip]


# new arguments
    path_modelgen = os.path.join(options.wd, options.cn, "model")
    CheckOrCreate(path_modelgen)
    options.dgtrain = os.path.join(path_modelgen, "data_generator_train.pkl")
    options.dgtest = os.path.join(path_modelgen, "data_generator_test.pkl")

    options.wd_32 = os.path.join(options.wd, options.cn, "FCN32")
    options.wd_16 = os.path.join(options.wd, options.cn, "FCN16")
    options.wd_8 = os.path.join(options.wd, options.cn, "FCN8")

    if options.loss == "weight" or options.loss == "weightcpp":
        Weight = True
        WeightOnes = False
    elif options.loss == "weight1" or options.loss == "weightcpp1":
        Weight = True
        WeightOnes = True
    else:
        Weight = False
        WeightOnes = False
    options.Weight = Weight
    options.WeightOnes = WeightOnes

    options.patients = ['141549', '572123', '581910',
                        '162438', '588626', '160120', '498959']

    transform_list = [Transf.Identity(),
                      Transf.Flip(0),
                      Transf.Flip(1)]

    for rot in np.arange(1, 360, 4):
        transform_list.append(Transf.Rotation(rot, enlarge=options.enlarge))

    for sig in [1, 2, 3, 4]:
        transform_list.append(Transf.OutOfFocus(sig))

    for i in range(50):
        transform_list.append(Transf.ElasticDeformation(0, 12, num_points=4))

    options.transform_list = transform_list

    return (options, args)
