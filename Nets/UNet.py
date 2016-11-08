from BasicNet import *
import os
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def unet(split, data_gene, loss, batch_size, Weight, cn, skip):
    n = caffe.NetSpec()
    if not Weight:
        n.data, n.label = DataLayer(split, data_gene, batch_size, cn, Weight)
    else:
        n.data, n.label, n.weight = DataLayer(
            split, data_gene, batch_size, cn, Weight)

    n.conv_d0a, n.relu_d0b = ConvRelu(n.data, 64)
    n.conv_d0b, n.relu_d0c = ConvRelu(n.relu_d0b, 64)
    n.pool_d0c = Maxpool(n.relu_d0c)

    n.conv_d1a, n.relu_d1b = ConvRelu(n.pool_d0c, 128)
    n.conv_d1b, n.relu_d1c = ConvRelu(n.relu_d1b, 128)
    n.pool_d1c = Maxpool(n.relu_d1c)

    n.conv_d2a, n.relu_d2b = ConvRelu(n.pool_d1c, 256)
    n.conv_d2b, n.relu_d2c = ConvRelu(n.relu_d2b, 256)
    n.pool_d2c = Maxpool(n.relu_d2c)

    n.conv_d3a, n.relu_d3b = ConvRelu(n.pool_d2c, 512)
    n.conv_d3b, n.relu_d3c = ConvRelu(n.relu_d3b, 512)
    n.drop_d3c = L.Dropout(n.relu_d3c, dropout_ratio=0.5, in_place=True)

    n.pool_d3c = Maxpool(n.drop_d3c)

    n.conv_d4a, n.relu_d4b = ConvRelu(n.pool_d3c, 512)
    n.conv_d4b, n.relu_d4c = ConvRelu(n.relu_d4b, 512)
    n.drop_d4c = L.Dropout(n.relu_d4c, dropout_ratio=0.5, in_place=True)

    if 4 not in skip:
        n.upconv_d4c_u3a, n.relu_u3a, n.crop_d3c, n.concat_d3cc_u3a, n.conv_u3b, n.relu_u3c, n.conv_u3c, n.relu_u3d = DeconvReCropConcatConvReConvRe(
            n.drop_d4c, n.drop_d3c, 512)
    else:
        n.upconv_d4c_u3a, n.relu_u3a, n.conv_u3b, n.relu_u3c, n.conv_u3c, n.relu_u3d = DeconvReConvReConvRe(
            n.drop_d4c, 512)
    if 3 not in skip:
        n.upconv_d3c_u2a, n.relu_u2a, n.crop_d2c, n.concat_d2cc_u2a, n.conv_u2b, n.relu_u2c, n.conv_u2c, n.relu_u2d = DeconvReCropConcatConvReConvRe(
            n.relu_u3d, n.relu_d2c, 256)
    else:
        n.upconv_d3c_u2a, n.relu_u2a, n.conv_u2b, n.relu_u2c, n.conv_u2c, n.relu_u2d = DeconvReConvReConvRe(
            n.relu_u3d, 256)
    if 2 not in skip:
        n.upconv_d2c_u1a, n.relu_u1a, n.crop_d1c, n.concat_d1cc_u1a, n.conv_u1b, n.relu_u1c, n.conv_u1c, n.relu_u1d = DeconvReCropConcatConvReConvRe(
            n.relu_u2d, n.relu_d1c, 128)
    else:
        n.upconv_d2c_u1a, n.relu_u1a, n.conv_u1b, n.relu_u1c, n.conv_u1c, n.relu_u1d = DeconvReConvReConvRe(
            n.relu_u2d, 128)
    if 1 not in skip:
        n.upconv_d1c_u0a, n.relu_u0a, n.crop_d0c, n.concat_d0cc_u0a, n.conv_u0b, n.relu_u0c, n.conv_u0c, n.relu_u0d = DeconvReCropConcatConvReConvRe(
            n.relu_u1d, n.relu_d0c, 64, deconv_out=128)
    else:
        n.upconv_d1c_u0a, n.relu_u0a, n.conv_u0b, n.relu_u0c, n.conv_u0c, n.relu_u0d = DeconvReConvReConvRe(
            n.relu_u1d, 64, deconv_out=128)

    n.score = L.Convolution(n.relu_u0d, kernel_size=1,
                            num_output=2, pad=0,
                            param=[dict(lr_mult=1, decay_mult=1), dict(
                                lr_mult=2, decay_mult=0)],
                            weight_filler=dict(type="xavier"))
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
    wd = options.wd
    skip = options.skip

    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(unet('train', dgtrain, loss, bs, wgt, cn, skip)))

    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(unet('test', dgtest, loss, 1, False, cn, skip)))
