
"""

line for kepler:

export WEIGHT=/data/users/pnaylor/Documents/Python/FCN/FCN2/temp_files/weights.pretrained.caffemodel
export WD=/data/users/pnaylor/Documents/Python/FCN
export RAWDATA=/data/users/pnaylor/Bureau/ToAnnotate

--rawdata $RAWDATA --wd $WD --cn FCN16 --weight $WEIGHT --niter 200 

"""


from DataToLMDB import MakeDataLikeFCN
import ImageTransf as Transf
import caffe
import os

from solver import solver
import FCN16
import numpy as np
import time

from optparse import OptionParser

import pdb


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


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

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--rawdata", dest="rawdata",
                      help="raw data folder, with respect to datamanager.py")

    parser.add_option("--wd", dest="wd",
                      help="Working directory")

    parser.add_option("--cn", dest="cn",
                      help="Classifier name, like FCN32")

    parser.add_option("--weight", dest="weight",
                      help="Where to find the weight file")

    parser.add_option("--niter", dest="niter",
                      help="Number of iterations")

    parser.add_option('--gpu', dest="gpu",
                      help="Which GPU to use.")
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
        options.cn = 'fcn16'

    if options.gpu is None:
        options.gpu = "0"

    print "Input paramters to run:"
    print " \n "
    print "Raw data direct   : | " + options.rawdata
    print "Work directory    : | " + options.wd
    print "Classifier name   : | " + options.cn
    print "Weight file (init): | " + options.weight
    print "Number of iteration | " + options.niter
    print "Which GPU         : | " + options.gpu

    options.niter = int(options.niter)

    create_dataset = False
    create_solver = True
    create_net = True

    enlarge = False  # create symetry if the image becomes black ?

    transform_list = [Transf.Identity(),
                      Transf.Rotation(45, enlarge=enlarge),
                      Transf.Rotation(90, enlarge=enlarge),
                      Transf.Rotation(135, enlarge=enlarge),
                      Transf.Flip(0),
                      Transf.Flip(1),
                      Transf.OutOfFocus(5),
                      Transf.OutOfFocus(10),
                      Transf.ElasticDeformation(0, 30, num_points=4),
                      Transf.ElasticDeformation(0, 30, num_points=4)]

    if create_dataset:
        MakeDataLikeFCN(options.rawdata, options.wd, transform_list)

    if create_net:
        data_train = options.wd  # os.path.join(options.wd, "train")
        data_test = options.wd  # os.path.join(options.wd, "test")
        CheckOrCreate(os.path.join(options.wd, options.cn))
        FCN16.make_net(os.path.join(options.wd, options.cn),
                       data_train, data_test,
                       classifier_name=options.cn,
                       classifier_name1="score_fr1")

    solver_path = os.path.join(options.wd, options.cn, "solver16.prototxt")
    outsnap = os.path.join(options.wd, options.cn, "snapshot", "snapshot")

    CheckOrCreate(os.path.join(options.wd, options.cn))
    CheckOrCreate(outsnap)

    if create_solver:
        name_solver = solver(solver_path,
                             os.path.join(options.wd, options.cn,
                                          "train.prototxt"),
                             test_net_path=os.path.join(
                                 options.wd, options.cn, "test.prototxt"),
                             base_lr=0.000000001,
                             out_snap=outsnap)
        # name_solver is solver_path.....
    weights = options.weight
    assert os.path.exists(weights)

    caffe.set_device(int(options.gpu))
    caffe.set_mode_gpu()

    niter = options.niter

    # pdb.set_trace()
    my_solver = caffe.get_solver(solver_path)
    # pdb.set_trace()
    my_solver.net.copy_from(weights)

    start_time = time.time()
    print 'Running solvers for %d iterations...' % niter

    solvers = [('pretrained', my_solver)]

    res_fold = os.path.join(options.wd, options.cn, "temp_files")
    loss, acc, weights = run_solvers(niter, solvers, res_fold)

    print 'Done.'

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image: (have to put number of images.. "
    diff_time = diff_time / 10
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)