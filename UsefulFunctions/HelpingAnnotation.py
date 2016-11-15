# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:01:58 2016

@author: naylorpeter
"""

import scipy.misc
import glob
from skimage.io import imread
from optparse import OptionParser
import os
import time


def GetNum(fil):
     num = fil[3:5]
     return num + ".png"


def Run(path, output, delete):
    if path[-1] != '/':
        path = path + "/"    
    if output[-1] != '/':
        output = output + "/"
    for f in glob.glob(path + '*.tif'):
        img_bin = imread(f) > 0
        f_n = output + f.split('/')[-1].split('.')[0] + ".png"
        scipy.misc.imsave(GetNum(f_n), img_bin)
    if delete == "1":
        for f in glob.glob(path + '*tif'):
            os.remove(f)
            
            
if __name__ == "__main__":
    
    parser = OptionParser()

    parser.add_option("-p", "--path", dest="path",
                      help="path to folder")
    parser.add_option("-o", "--output_folder", dest="out",default='default',
                      help="Output folder, if unassigned is the path folder")
    parser.add_option('-d', '--del', dest="delete",default="0",
                      help="delete old files? 1:yes 0:no")
                      
    (options, args) = parser.parse_args()

    if options.out == "default":
        options.out = options.path
    
    print "Input paramters to HelpingAnnotation:"
    print " \n "
    print "Path              : | " + options.path
    print "Output folder     : | " + options.out
    print "Delete?           : | " + options.delete
    
    if not os.path.isdir(options.out):
            os.mkdir(options.out)
    
    print ' \n '       
    print "Beginning analyse:" 
    
    start_time = time.time()
### Core of the code    
########################################
    
    Run(options.path, options.out, options.delete)

########################################
    
    diff_time = time.time() - start_time

    print ' \n '
    print 'Time:'
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
    