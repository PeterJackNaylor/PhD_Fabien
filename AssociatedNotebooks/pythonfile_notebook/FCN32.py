# -*- coding: utf-8 -*-


## cell 1 
from DataToLMDB import MakeLMDB
import ImageTransf as Transf

path = '/home/naylor/Bureau/ToAnnotate'
wd = '/home/naylor/Documents/Python/PhD/PhD_Fabien/AssociatedNotebooks'

output_dir = path + '/lmdb'

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

mean_ = MakeLMDB(path, output_dir, transform_list, val_num = 2, get_mean=True, verbose = False)


## cell 2

import sys
import caffe
import os

caffe.set_device(0)
caffe.set_mode_gpu()

weights =  '/home/naylor/Documents/FCN/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel'
assert os.path.exists(weights)
solver = None


## cell 3 


import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn32(batch_size, lmdb, mean_, layer_name = ""):
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             scale = 1/256., ntop=2)
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    
    n.score_fr = L.Convolution(n.drop7, num_output=2, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    
    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=2, kernel_size=64, stride=32,
            bias_term=False),
        param=[dict(lr_mult=0)])
    
    n.score = crop(n.upscore, n.data)
    
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()

def make_net():
    if not os.path.isdir('./fcn32'):
        os.mkdir('./fcn32')
    with open('fcn32/train.prototxt', 'w') as f:
        f.write(str(fcn32(5, output_dir, mean_)))


if __name__ == '__main__':
    make_net()
    # load the solver
    #solver = caffe.SGDSolver('fcn32/train.prototxt')