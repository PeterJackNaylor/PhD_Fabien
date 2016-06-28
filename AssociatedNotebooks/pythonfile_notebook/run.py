# -*- coding: utf-8 -*-


from DataToLMDB import MakeLMDB

import ImageTransf as Transf
import caffe
import os

from solver import solver
import FCN32
import numpy as np
import time

from optparse import OptionParser

import pdb

def run_solvers(niter, solvers, res_fold, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
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

    parser.add_option("-d", "--data", dest="data",
                      help="Input file (raw data)")
    parser.add_option("--wd", dest="wd",
                      help="Working directory")
    parser.add_option("--rawdata", dest="rawdata", 
                      help="raw data folder, with respect to datamanager.py")
    parser.add_option("--outputfolder", dest="outputfolder",
                      help="Where the temporay weight folder will be stored")
    parser.add_option("--cn", dest="cn", 
                      help="Classifier name, like FCN32")
    parser.add_option("--weight", dest="weight", 
                      help="Where to find the weight file")
    parser.add_option("--niter", dest="niter", 
                      help="Number of iterations")    
    
    (options, args) = parser.parse_args()
    
    if options.rawdata is None:
        options.rawdata = '/home/naylor/Bureau/ToAnnotate'
        
    if options.wd is None:
        options.wd = '/home/naylor/Documents/Python/PhD/PhD_Fabien/AssociatedNotebooks/pythonfile_notebook/'
        
    if options.data is None:
        options.data = "lmdb"
        
    if options.weight is None:
        options.weight = '/home/naylor/Documents/FCN/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel'
        
    if options.niter is None:
        options.niter = "200"
    
    if options.cn is None:
        options.cn = 'fcn32'
        
    if options.outputfolder is None:
        options.outputfolder = "temp_weight"
        
    print "Input paramters to CuttingPatches:"
    print " \n "
    print "Work directory    : | " + options.wd
    print "Name of data(lmdb): | " + options.data
    print "Raw data direct   : | " + options.rawdata
    print "Classifier name   : | " + options.cn
    print "Weight file (init): | " + options.weight
    print "Weight files (out): | " + options.outputfolder    
    print "Number of iteration | " + options.niter
    
        
    options.niter = int(options.niter)
    
    create_dataset = False
    create_solver = False
    create_net = False
    
    output_dir = options.wd + options.data

    enlarge = False ## create symetry if the image becomes black ? 

    transform_list = [Transf.Identity(),
                  Transf.Rotation(45, enlarge=enlarge), 
                  Transf.Rotation(90, enlarge=enlarge),
                  Transf.Rotation(135, enlarge=enlarge),
                  Transf.Flip(0),
                  Transf.Flip(1),
                  Transf.OutOfFocus(5),
                  Transf.OutOfFocus(10),
                  Transf.ElasticDeformation(0, 30, num_points = 4),
                  Transf.ElasticDeformation(0, 30, num_points = 4)]
    
    if create_dataset:
        
        mean_ = MakeLMDB(options.rawdata, output_dir, transform_list, val_num = 2, get_mean=True, verbose = False)
    
    batchsize = 10
    data_folder = output_dir
    output_folder = options.wd + options.cn
     
    if create_net:
        
   
        FCN32.make_net(batchsize, data_folder, output_folder)

    solver_path = options.wd + options.cn + "/solver.protoxt"
    
    if create_solver:
        
        solver(solver_path, test_net_path=None, base_lr=0.001, out_snap = options.wd + options.cn + "/temp_snapshot")
     
    weights =  options.weight
    assert os.path.exists(weights)
        
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    niter = options.niter
    
    pdb.set_trace()
    my_solver = caffe.get_solver(solver_path)
    my_solver.net.copy_from(weights)
    

    
    start_time = time.time()
    print 'Running solvers for %d iterations...' % niter
    
    
    solvers = [('pretrained', my_solver)]
               
    res_fold = options.wd + "/" + options.outputfolder
    loss, acc, weights = run_solvers(niter, solvers, res_fold)
    
    print 'Done.'
        
        
    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
    
    print ' \n '
    print "Average time per image: (have to put number of images.. " 
    diff_time = diff_time / 10
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)            
        
        
        