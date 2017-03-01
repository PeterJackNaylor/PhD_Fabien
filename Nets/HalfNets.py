from BasicNet import *
import os
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def MakeHalfDeconvNet(options):
    dgtrain = options.dgtrain
    dgtest = options.dgtest
    cn = options.cn
    loss = options.loss
    bs = options.batch_size
    path = os.path.join(options.wd, options.cn)
    num_output = options.num_output
    with open(os.path.join(path, 'train.prototxt'), 'w') as f:
        f.write(str(HalfDeconvNet('train', dgtrain, loss, bs, cn, num_output)))

    with open(os.path.join(path, 'test.prototxt'), 'w') as f:
        f.write(str(HalfDeconvNet('test', dgtest, loss, 1, cn, num_output)))



def HalfDeconvNet(split, data_gene, loss, batch_size, cn, num_output):
    n = caffe.NetSpec()

    n.data, n.label = DataLayer(split, data_gene, batch_size, cn, False)

    n.conv1_1, n.BatchNormalize1_1, n.scaler1_1, n.relu1_1 = ConvBnRelu(
        n.data, 64, 3, 1, 1)
    n.conv1_2, n.BatchNormalize1_2, n.scaler1_2, n.relu1_2 = ConvBnRelu(
        n.relu1_1, 64, 3, 1, 1)

    n.pool1, n.pool1_mask = Maxpool(
        n.relu1_2, ks=2, stride=2, ntop=2)

    n.conv2_1, n.BatchNormalize2_1, n.scaler2_1, n.relu2_1 = ConvBnRelu(
        n.pool1, 128, 3, 1, 1)
    n.conv2_2, n.BatchNormalize2_2, n.scaler2_2, n.relu2_2 = ConvBnRelu(
        n.relu2_1, 128, 3, 1, 1)

    n.pool2, n.pool2_mask = Maxpool(
        n.relu2_2, ks=2, stride=2, ntop=2)

    n.conv3_1, n.BatchNormalize3_1, n.scaler3_1, n.relu3_1 = ConvBnRelu(
        n.pool2, 256, 3, 1, 1)
    n.conv3_2, n.BatchNormalize3_2, n.scaler3_2, n.relu3_2 = ConvBnRelu(
        n.relu3_1, 256, 3, 1, 1)
    n.conv3_3, n.BatchNormalize3_3, n.scaler3_3, n.relu3_3 = ConvBnRelu(
        n.relu3_2, 256, 3, 1, 1)

    n.pool3, n.pool3_mask = Maxpool(
        n.relu3_3, ks=2, stride=2, ntop=2)

    n.conv4_1, n.BatchNormalize4_1, n.scaler4_1, n.relu4_1 = ConvBnRelu(
        n.pool3, 512, 3, 1, 1)
    n.conv4_2, n.BatchNormalize4_2, n.scaler4_2, n.relu4_2 = ConvBnRelu(
        n.relu4_1, 512, 3, 1, 1)
    n.conv4_3, n.BatchNormalize4_3, n.scaler4_3, n.relu4_3 = ConvBnRelu(
        n.relu4_2, 512, 3, 1, 1)

    n.pool4, n.pool4_mask = Maxpool(
        n.relu4_3, ks=2, stride=2, ntop=2)

    n.conv5_1, n.BatchNormalize5_1, n.scaler5_1, n.relu5_1 = ConvBnRelu(
        n.pool4, 512, 3, 1, 1)
    n.conv5_2, n.BatchNormalize5_2, n.scaler5_2, n.relu5_2 = ConvBnRelu(
        n.relu5_1, 512, 3, 1, 1)
    n.conv5_3, n.BatchNormalize5_3, n.scaler5_3, n.relu5_3 = ConvBnRelu(
        n.relu5_2, 512, 3, 1, 1)

    n.pool5, n.pool5_mask = Maxpool(
        n.relu5_3, ks=2, stride=2, ntop=2)

    n.fc6, n.BatchNormalize6, n.scaler6_1, n.relu6 = ConvBnRelu(
        n.pool5, 4096, ks=7, stride=1, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7, n.BatchNormalize7, n.scaler7_1, n.relu7 = ConvBnRelu(
        n.relu6, 4096, ks=1, stride=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score = L.InnerProduct(n.drop7, num_output=num_output, weight_filler=xavier)

    n.loss = LossLayer(n.score, n.label, loss)

    return n.to_proto()


def HalfUNet(split, data_gene, loss, batch_size, Weight, cn, skip, num_output):
    n = caffe.NetSpec()
    if not Weight:
        n.data, n.label = DataLayer(split, data_gene, batch_size, cn, Weight)
    else:
        n.data, n.label, n.weight = DataLayer(
            split, data_gene, batch_size, cn, Weight)

    n.conv_d0a, n.relu_d0b = ConvRelu(n.data, 64, pad=0)
    n.conv_d0b, n.relu_d0c = ConvRelu(n.relu_d0b, 64, pad=0)
    n.drop_d0c = L.Dropout(n.relu_d0c, dropout_ratio=0.5, in_place=True)
    n.pool_d0c = Maxpool(n.drop_d0c)

    n.conv_d1a, n.relu_d1b = ConvRelu(n.pool_d0c, 128, pad=0)
    n.conv_d1b, n.relu_d1c = ConvRelu(n.relu_d1b, 128, pad=0)
    n.drop_d1c = L.Dropout(n.relu_d1c, dropout_ratio=0.5, in_place=True)
    n.pool_d1c = Maxpool(n.drop_d1c)

    n.conv_d2a, n.relu_d2b = ConvRelu(n.pool_d1c, 256, pad=0)
    n.conv_d2b, n.relu_d2c = ConvRelu(n.relu_d2b, 256, pad=0)
    n.drop_d2c = L.Dropout(n.relu_d2c, dropout_ratio=0.5, in_place=True)
    n.pool_d2c = Maxpool(n.drop_d2c)

    n.conv_d3a, n.relu_d3b = ConvRelu(n.pool_d2c, 512, pad=0)
    n.conv_d3b, n.relu_d3c = ConvRelu(n.relu_d3b, 512, pad=0)
    n.drop_d3c = L.Dropout(n.relu_d3c, dropout_ratio=0.5, in_place=True)

    n.pool_d3c = Maxpool(n.drop_d3c)

    n.conv_d4a, n.relu_d4b = ConvRelu(n.pool_d3c, 512, pad=0)
    n.conv_d4b, n.relu_d4c = ConvRelu(n.relu_d4b, 512, pad=0)
    n.drop_d4c = L.Dropout(n.relu_d4c, dropout_ratio=0.5, in_place=True)


    return n.to_proto()
