from BasicNet import *
import os
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def deconvnet(split, data_gene, loss, batch_size, Weight, cn):
    n = caffe.NetSpec()
    if not Weight:
        n.data, n.label = DataLayer(split, data_gene, batch_size, cn, Weight)
    else:
        n.data, n.label, n.weight = DataLayer(
            split, data_gene, batch_size, cn, Weight)

    n.conv1_1, n.BatchNormalize1_1, n.scaler1_1, n.relu1_1 = ConvBnRelu(
        n.data, 64, 3, 1, 1)
    n.conv1_2, n.BatchNormalize1_2, n.scaler1_2, n.relu1_2 = ConvBnRelu(
        n.relu1_1, 64, 3, 1, 1)

    n.pool1, n.pool1_mask, n.pool1_count = Maxpool(
        n.relu1_2, ks=2, stride=2, ntop=3)

    n.conv2_1, n.BatchNormalize2_1, n.scaler2_1, n.relu2_1 = ConvBnRelu(
        n.pool1, 128, 3, 1, 1)
    n.conv2_2, n.BatchNormalize2_2, n.scaler2_2, n.relu2_2 = ConvBnRelu(
        n.relu2_1, 128, 3, 1, 1)

    n.pool2, n.pool2_mask, n.pool2_count = Maxpool(
        n.relu2_2, ks=2, stride=2, ntop=3)

    n.conv3_1, n.BatchNormalize3_1, n.scaler3_1, n.relu3_1 = ConvBnRelu(
        n.pool2, 256, 3, 1, 1)
    n.conv3_2, n.BatchNormalize3_2, n.scaler3_2, n.relu3_2 = ConvBnRelu(
        n.relu3_1, 256, 3, 1, 1)
    n.conv3_3, n.BatchNormalize3_3, n.scaler3_3, n.relu3_3 = ConvBnRelu(
        n.relu3_2, 256, 3, 1, 1)

    n.pool3, n.pool3_mask, n.pool3_count = Maxpool(
        n.relu3_3, ks=2, stride=2, ntop=3)

    n.conv4_1, n.BatchNormalize4_1, n.scaler4_1, n.relu4_1 = ConvBnRelu(
        n.pool3, 512, 3, 1, 1)
    n.conv4_2, n.BatchNormalize4_2, n.scaler4_2, n.relu4_2 = ConvBnRelu(
        n.relu4_1, 512, 3, 1, 1)
    n.conv4_3, n.BatchNormalize4_3, n.scaler4_3, n.relu4_3 = ConvBnRelu(
        n.relu4_2, 512, 3, 1, 1)

    n.pool4, n.pool4_mask, n.pool4_count = Maxpool(
        n.relu4_3, ks=2, stride=2, ntop=3)

    n.conv5_1, n.BatchNormalize5_1, n.scaler5_1, n.relu5_1 = ConvBnRelu(
        n.pool4, 512, 3, 1, 1)
    n.conv5_2, n.BatchNormalize5_2, n.scaler5_2, n.relu5_2 = ConvBnRelu(
        n.relu5_1, 512, 3, 1, 1)
    n.conv5_3, n.BatchNormalize5_3, n.scaler5_3, n.relu5_3 = ConvBnRelu(
        n.relu5_2, 512, 3, 1, 1)

    n.pool5, n.pool5_mask, n.pool5_count = Maxpool(
        n.relu5_3, ks=2, stride=2, ntop=3)

    n.fc6, n.BatchNormalize6, n.scaler6_1, n.relu6 = ConvBnRelu(
        n.pool5, 4096, ks=7, stride=1, pad=0)
    #n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7, n.BatchNormalize7, n.scaler7_1, n.relu7 = ConvBnRelu(
        n.relu6, 4096, ks=1, stride=1, pad=0)
    #n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # no need for dropout as we have batch normalization!
    n.deconv_fc6, n.deconv_fc6_bn, n.descaler6_1, n.deconv_fc6_relu = DeconvBnRelu(
        n.relu7, 512, ks=7)

    n.unpool5 = Maxunpool(n.deconv_fc6_relu, n.pool5_mask,
                          n.pool5_count)

    n.deconv5_1, n.DeBatchNormalize5_1, n.descaler5_1, n.derelu5_1 = DeconvBnRelu(
        n.unpool5, 512, ks=3, pad=1)
    n.deconv5_2, n.DeBatchNormalize5_2, n.descaler5_2, n.derelu5_2 = DeconvBnRelu(
        n.derelu5_1, 512, ks=3, pad=1)
    n.deconv5_3, n.DeBatchNormalize5_3, n.descaler5_3, n.derelu5_3 = DeconvBnRelu(
        n.derelu5_2, 512, ks=3, pad=1)

    n.unpool4 = Maxunpool(n.derelu5_3, n.pool4_mask,
                          n.pool4_count)

    n.deconv4_1, n.DeBatchNormalize4_1, n.descaler4_1, n.derelu4_1 = DeconvBnRelu(
        n.unpool4, 512, ks=3, pad=1)
    n.deconv4_2, n.DeBatchNormalize4_2, n.descaler4_2, n.derelu4_2 = DeconvBnRelu(
        n.derelu4_1, 512, ks=3, pad=1)
    n.deconv4_3, n.DeBatchNormalize4_3, n.descaler4_3, n.derelu4_3 = DeconvBnRelu(
        n.derelu4_2, 256, ks=3, pad=1)

    n.unpool3 = Maxunpool(n.derelu4_3, n.pool3_mask,
                          n.pool3_count)

    n.deconv3_1, n.DeBatchNormalize3_1, n.descaler3_1, n.derelu3_1 = DeconvBnRelu(
        n.unpool3, 256, ks=3, pad=1)
    n.deconv3_2, n.DeBatchNormalize3_2, n.descaler3_2, n.derelu3_2 = DeconvBnRelu(
        n.derelu3_1, 256, ks=3, pad=1)
    n.deconv3_3, n.DeBatchNormalize3_3, n.descaler3_3, n.derelu3_3 = DeconvBnRelu(
        n.derelu3_2, 128, ks=3, pad=1)

    n.unpool2 = Maxunpool(n.derelu3_3, n.pool2_mask,
                          n.pool2_count)

    n.deconv2_1, n.DeBatchNormalize2_1, n.descaler2_1, n.derelu2_1 = DeconvBnRelu(
        n.unpool2, 128, ks=3, pad=1)
    n.deconv2_2, n.DeBatchNormalize2_2, n.descaler2_2, n.derelu2_2 = DeconvBnRelu(
        n.derelu2_1, 64, ks=3, pad=1)

    n.unpool1 = Maxunpool(n.derelu2_2, n.pool1_mask,
                          n.pool1_count)

    n.deconv1_1, n.DeBatchNormalize1_1, n.descaler1_1, n.derelu1_1 = DeconvBnRelu(
        n.unpool1, 64, ks=3, pad=1)
    n.deconv1_2, n.DeBatchNormalize1_2, n.descaler1_2, n.derelu1_2 = DeconvBnRelu(
        n.derelu1_1, 64, ks=3, pad=1)

    n.score = L.Convolution(n.derelu1_2, kernel_size=3, stride=1,
                            num_output=2, pad=1,
                            param=[dict(lr_mult=1, decay_mult=1), dict(
                                lr_mult=2, decay_mult=0)],
                            weight_filler=Gaussian_fil,
                            bias_filler=Constant_fil)

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
        f.write(str(deconvnet('train', dgtrain, loss, bs, wgt, cn)))

    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(deconvnet('test', dgtest, loss, 1, False, cn)))
