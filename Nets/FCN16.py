from BasicNet import *
import os
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def fcn16(split, data_gene, loss, batch_size, Weight, cn, c1):
    n = caffe.NetSpec()
    if not Weight:
        n.data, n.label = DataLayer(split, data_gene, batch_size, cn, Weight)
    else:
        n.data, n.label, n.weight = DataLayer(
            split, data_gene, batch_size, cn, Weight)

    n.conv1_1, n.relu1_1 = ConvRelu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = ConvRelu(n.relu1_1, 64)
    n.pool1 = Maxpool(n.relu1_2)

    n.conv2_1, n.relu2_1 = ConvRelu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = ConvRelu(n.relu2_1, 128)
    n.pool2 = Maxpool(n.relu2_2)

    n.conv3_1, n.relu3_1 = ConvRelu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = ConvRelu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = ConvRelu(n.relu3_2, 256)
    n.pool3 = Maxpool(n.relu3_3)

    n.conv4_1, n.relu4_1 = ConvRelu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = ConvRelu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = ConvRelu(n.relu4_2, 512)
    n.pool4 = Maxpool(n.relu4_3)

    n.conv5_1, n.relu5_1 = ConvRelu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = ConvRelu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = ConvRelu(n.relu5_2, 512)
    n.pool5 = Maxpool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6 = ConvRelu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7, n.relu7 = ConvRelu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    score_fr = Conv(n.drop7, nout=2, ks=1, pad=0)
    n.__setattr__(c1, score_fr)

    n.upscore2 = L.Deconvolution(score_fr,
                                 convolution_param=dict(num_output=2, kernel_size=4, stride=2,
                                                        bias_term=False),
                                 weight_filler=dict(type='bilinear'),
                                 param=[dict(lr_mult=1)])
    n.__setattr__(c2, upscore2)

    n.score_pool4 = Conv(n.pool4, nout=2, ks=1, pad=0)

    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
                             operation=P.Eltwise.SUM)
    n.upscore16 = L.Deconvolution(n.fuse_pool4,
                                  convolution_param=dict(num_output=2, kernel_size=32, stride=16,
                                                         bias_term=False),
                                  weight_filler=dict(type='bilinear'),
                                  param=[dict(lr_mult=1)])

    n.score = crop(n.upscore16, n.data)

    if not Weight:
        n.loss = LossLayer(n.score, n.label, loss)
    else:
        n.loss = LossLayer(n.score, n.label, loss, weight=n.weight)
    return n.to_proto()


def make_net(options, c1="score_fr_32", c2="upscore_16"):
    dgtrain = options.dgtrain
    dgtest = options.dgtest
    cn = options.cn
    loss = options.loss
    bs = options.batch_size
    wgt = options.Weight
    wd = options.wd_16

    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(fcn16('train', dgtrain, loss, bs, wgt, cn, c1)))

    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(fcn16('test', dgtest, loss, 1, False, cn, c1)))
