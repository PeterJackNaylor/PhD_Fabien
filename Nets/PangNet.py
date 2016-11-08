from BasicNet import *
import os
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def pangnet(split, data_gene, loss, batch_size, Weight, cn):
    n = caffe.NetSpec()
    if not Weight:
        n.data, n.label = DataLayer(split, data_gene, batch_size, cn, Weight)
    else:
        n.data, n.label, n.weight = DataLayer(
            split, data_gene, batch_size, cn, Weight)

    n.conv1, n.relu1 = conv_relu(n.data, 8, pad=1)
    n.conv2, n.relu2 = conv_relu(n.relu1, 8, pad=1)
    n.conv3, n.relu3 = conv_relu(n.relu2, 8, pad=1)

    n.score = Conv(n.relu3, nout=2, ks=1, pad=0)

    if not Weight:
        n.loss = LossLayer(n.score, n.label, loss)
    else:
        n.loss = LossLayer(n.score, n.label, loss, weight=n.weight)
    return n.to_proto()


def make_net(options):
    dgtrain = options.datagen_path_train
    dgtest = options.datagen_path_test
    cn = options.cn
    loss = options.loss
    bs = options.batch_size
    wgt = options.Weight
    wd = options.wd

    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(pangnet('train', dgtrain, loss, bs, wgt, cn)))

    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(pangnet('test', dgtest, loss, 1, False, cn)))
