
import sys

sys.path.append(
    "/data/users/pnaylor/Documents/Python/PhD_Fabien/UsefulFunctions/")
sys.path[4] = "/data/users/pnaylor/Documents/Python/caffe_peter/python"


def switch_caffe_path():
    sys_path = [el for el in sys.path if 'caffe' not in el]
    caffe_path = [el for el in sys.path if 'caffe' in el][0]

    caffe_path = caffe_path.replace("/caffe/", "/caffe_deconv/")
    sys.path = sys_path + [caffe_path]


# switch_caffe_path()

import caffe
from caffe import layers as L, params as P
import os
import pdb
from caffe.coord_map import crop
import sys


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1),
                                dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type="xavier"))
    return conv, L.ReLU(conv, in_place=True)


def Deconv(bottom, nout, ks, pad, weight_filler, bias_filler):
    deconv = L.Deconvolution(bottom,
                             param=[dict(lr_mult=1, decay_mult=1),
                                    dict(lr_mult=2, decay_mult=0)],
                             convolution_param=dict(num_output=nout,
                                                    pad=pad,
                                                    kernel_size=ks,
                                                    weight_filler=weight_filler,
                                                    bias_filler=bias_filler))
    return deconv

# had to improvise with respect to bn_mode=INFERENCE


def DeconvReCropConcatConvReConvRe(bottom1, bridge2, val, deconv_out=None):
    if deconv_out is None:
        deconv_out = val

    deconv = L.Deconvolution(bottom1,
                             convolution_param=dict(num_output=deconv_out, kernel_size=2, stride=2,
                                                    bias_term=False, weight_filler=dict(type="xavier")),
                             param=[dict(lr_mult=1, decay_mult=1),
                                    dict(lr_mult=2, decay_mult=0)],
                             )
    relu1 = Relu(deconv)
    crop = crop(bridge2, relu1)
    concat = L.Concat(relu1, crop)
    conv2, relu2 = conv_relu(concat, val)
    conv3, relu3 = conv_relu(conv2, val)

    return deconv, relu1, crop, concat, conv2, relu2, conv3, relu3


def BatchNormalizer(bottom):
    noth = dict(lr_mult=0)
    param = [noth, noth, noth]
    bn = L.BatchNorm(bottom, param=param, in_place=True)
    return bn


def BatchNormalizer(bottom):
    noth = dict(lr_mult=0)
    param = [noth, noth, noth]
    bn = L.BatchNorm(bottom, param=param, in_place=True)
    sl = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True)
    return bn, sl


def Relu(bottom):
    return L.ReLU(bottom, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    layer = L.Pooling(bottom, pool=P.Pooling.MAX,
                      kernel_size=ks, stride=stride)
    return layer


def max_unpool(bottom1, bottom2, bottom3, ks=2, stride=2):
    unpooling_param = dict(pool=P.Pooling.MAX, kernel_size=ks,
                           stride=stride)
    # should be unpooling?
    return L.Unpooling(bottom1, bottom2, bottom3, pooling_param=unpooling_param)


def UNet(split, data_gene, classifier_name="UNet"):
    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
                         seed=1337, batch_size=batch_size, classifier_name=classifier_name)
    pylayer = 'DataLayerPeter'
    pydata_params["datagen"] = data_gene
    n.data, n.label = L.Python(module='DataLayerPeter', layer=pylayer,
                               ntop=2, param_str=str(pydata_params))

    n.conv_d0a, n.relu_d0b = conv_relu(n.data, 64)
    n.conv_d0b, n.relu_d0c = conv_relu(n.relu_d0b, 64)
    n.pool_d0c = max_pool(n.relu_d0c)

    n.conv_d1a, n.relu_d1b = conv_relu(n.pool_d0c, 128)
    n.conv_d1b, n.relu_d1c = conv_relu(n.relu_d1b, 128)
    n.pool_d1c = max_pool(n.relu_d1c)

    n.conv_d2a, n.relu_d2b = conv_relu(n.pool_d1c, 256)
    n.conv_d2b, n.relu_d2c = conv_relu(n.relu_d2b, 256)
    n.pool_d2c = max_pool(n.relu_d2c)

    n.conv_d3a, n.relu_d3b = conv_relu(n.pool_d2c, 512)
    n.conv_d3b, n.relu_d3c = conv_relu(n.relu_d3b, 512)
    n.drop_d3c = L.Dropout(n.relu_d3c, dropout_ratio=0.5, in_place=True)

    n.pool_d3c = max_pool(n.drop_d3c)

    n.conv_d4a, n.relu_d4b = conv_relu(n.pool_d3c, 512)
    n.conv_d4b, n.relu_d4c = conv_relu(n.relu_d4b, 512)
    n.drop_d4c = L.Dropout(n.relu_d4c, dropout_ratio=0.5, in_place=True)

    n.upconv_d4c_u3a, n.relu_u3a, n.crop_d3c, n.concat_d3cc_u3a, n.conv_u3b, n.relu_u3c, n.conv_u3c, n.relu_u3d = DeconvReCropConcatConvReConvRe(
        n.drop_d4c, n.drop_d3c, 512)

    n.upconv_d3c_u2a, n.relu_u2a, n.crop_d2c, n.concat_d2cc_u2a, n.conv_u2b, n.relu_u2c, n.conv_u2c, n.relu_u2d = DeconvReCropConcatConvReConvRe(
        n.relu_u3d, n.relu_d2c, 256)

    n.upconv_d2c_u1a, n.relu_u1a, n.crop_d1c, n.concat_d1cc_u1a, n.conv_u1b, n.relu_u1c, n.conv_u1c, n.relu_u1d = DeconvReCropConcatConvReConvRe(
        n.relu_u2d, n.relu_d1c, 128)

    n.upconv_d1c_u0a, n.relu_u0a, n.crop_d0c, n.concat_d0cc_u0a, n.conv_u0b, n.relu_u0c, n.conv_u0c, n.relu_u0d = DeconvReCropConcatConvReConvRe(
        n.relu_u1d, n.relu_d0c, 64, deconv_out=128)

    n.score = L.Convolution(n.relu_u0d, kernel_size=1,
                            num_output=2, pad=0,
                            param=[dict(lr_mult=1, decay_mult=1), dict(
                                lr_mult=2, decay_mult=0)],
                            weight_filler=dict(type="xavier"))

    if split != "val":
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
                                   loss_param=dict(normalize=False, ignore_label=255))
    return n.to_proto()


def make_net(wd, data_gene_train, data_gene_test, classifier_name="UNet"):
    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(UNet('train', data_gene_train, batch_size, classifier_name)))
    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(UNet('test', data_gene_test, 1, classifier_name)))
