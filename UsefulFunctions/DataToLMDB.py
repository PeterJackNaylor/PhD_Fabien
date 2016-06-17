# -*- coding: utf-8 -*-

import caffe
import lmdb
import numpy as np
import os
import Datamanager as Dm
import ImageTransf as Transf
from optparse import OptionParser
import time
import pdb


def MakeLMDB(path, output_dir, transform_list,
             img_width = 512, img_height = 512,
             val_num = 2, count = False,
             verbose = True
             ):
    
    test = Dm.DataManager(path)
    test.prepare_sets(leave_out = val_num)
    
    test.SetTransformation(transform_list)    
    
    color_lmdb_name = output_dir + '/color-lmdb'
    if not os.path.isdir(color_lmdb_name):
        os.makedirs(color_lmdb_name)
    color_in_db = lmdb.open(color_lmdb_name, map_size=int(1e12))
    
    label_lmdb_name = output_dir + '/label-lmdb'
    if not os.path.isdir(label_lmdb_name):
        os.makedirs(label_lmdb_name)
    label_in_db = lmdb.open(label_lmdb_name, map_size=int(1e12))
    
    num_images = 0;
    color_mean_color = np.zeros((3))
    
    img_width = 512
    img_height = 512
    
    in_idx = 0 
    num_img = (len(test.nii) - val_num) * len(transform_list)
    with color_in_db.begin(write=True) as color_in_txn:
        with label_in_db.begin(write=True) as label_in_txn:
            for img, img_gt, name in test.TrainingIteratorLeaveValOut():
                
                
                if verbose:
                    print(str(in_idx + 1) + ' / ' + str(num_img))
    
                # load image
                im = img
                

                assert im.dtype == np.uint8
                # RGB to BGR
                #pdb.set_trace()
                im = im[:,:,::-1]
                # in Channel x Height x Width order (switch from H x W x C)
                im = im.transpose((2,0,1))
    
                # compute mean color image
                for i in range(3):
                    color_mean_color[i] += im[i,:,:].mean()
                num_images += 1
    
                #color_im_dat = caffe.io.array_to_datum(im)
                color_im_dat = caffe.proto.caffe_pb2.Datum()
                color_im_dat.channels, color_im_dat.height, color_im_dat.width = im.shape
                assert color_im_dat.height == img_height
                assert color_im_dat.width == img_width
                color_im_dat.data = im.tostring()
                color_in_txn.put('{:0>12d}'.format(in_idx), color_im_dat.SerializeToString())
                
                #pdb.set_trace()
                im_gt = img_gt
                if im_gt.dtype != np.uint8:
                    pdb.set_trace()
                assert im_gt.dtype == np.uint8
                label_im_dat = caffe.proto.caffe_pb2.Datum()
                label_im_dat.channels = 1
                if len(img_gt.shape) == 2:
                    label_im_dat.height, label_im_dat.width = im_gt.shape
                else:
                    label_im_dat.height, label_im_dat.width, label_im_dat.channels = im_gt.shape
                assert label_im_dat.height == img_height
                assert label_im_dat.width == img_width
                label_im_dat.data = im.tostring()
                label_in_txn.put('{:0>12d}'.format(in_idx), label_im_dat.SerializeToString())
                
                in_idx += 1
    if count:
        return in_idx

if __name__ == "__main__":
    
    parser = OptionParser()

    parser.add_option("-p", "--path", dest="path",
                      help="Input path (raw data)")
    parser.add_option("-o", "--out", dest="out",
                      help="Where to store the lmdb files")
    parser.add_option('-n', '--leaveout', dest="leaveout",default="2",
                      help="Resolution")
    parser.add_option('-w', '--width', dest="width",default="512",
                      help="width")
    parser.add_option('--heigth', dest="heigth",default="512",
                      help="heigth")
    (options, args) = parser.parse_args()
              
    
    if options.path is None:
        path = '/home/naylor/Bureau/ToAnnotate'
    else:
        path = options.path
    
    if options.out is None:
        output_dir = path + '/lmdb'
    else:
        output_dir = options.out


    print " \n "
    print "Input paramters to DataToLMDB:"
    print " \n "
    print "Input file        : | " + path
    print "Output folder     : | " + output_dir
    print "Leave out         : | " + options.leaveout
    print "Heigth            : | " + options.heigth
    print "Width             : | " + options.width
    
    
    val_num = int(options.leaveout)
    enlarge = False
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
    
    print ' \n '       
    print "Beginning analyse:" 
    
    start_time = time.time()
### Core of the code    
########################################
    
   
    number = MakeLMDB(path, output_dir,
                      transform_list, 
                      img_width = int(options.width),
                      img_height = int(options.heigth),
                      val_num = val_num,
                      count= True)
########################################
    
    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for all:'
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
    
    print ' \n '
    print "Average time per image:" 
    diff_time = diff_time / number
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
    

    
    
                          
