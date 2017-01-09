from BasicNet import *
import os
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def pangnet(split, data_gene, loss, batch_size, Weight, cn, num_output):
    n = caffe.NetSpec()
    if not Weight:
        n.data, n.label = DataLayer(split, data_gene, batch_size, cn, Weight)
    else:
        n.data, n.label, n.weight = DataLayer(
            split, data_gene, batch_size, cn, Weight)

    n.conv1, n.relu1 = ConvRelu(n.data, 8, pad=1)
    n.conv2, n.relu2 = ConvRelu(n.relu1, 8, pad=1)
    n.conv3, n.relu3 = ConvRelu(n.relu2, 8, pad=1)

    n.score = Conv(n.relu3, nout=num_output, ks=1, pad=0)

    if not Weight:
        n.loss = LossLayer(n.score, n.label, loss)
    else:
        n.loss = LossLayer(n.score, n.label, loss, weight=n.weight)
    return n.to_proto()


def make_net(options):
    dgtrain = options.dgtrain
    dgtest = options.dgtest
    cn = options.cn
    loss = options.loss
    bs = options.batch_size
    wgt = options.Weight
    path = os.path.join(options.wd, options.cn)
    num_output = options.num_output
    with open(os.path.join(path, 'train.prototxt'), 'w') as f:
        f.write(str(pangnet('train', dgtrain, loss, bs, wgt, cn, num_output)))

    with open(os.path.join(path, 'test.prototxt'), 'w') as f:
        f.write(str(pangnet('test', dgtest, loss, 1, False, cn, num_output)))
