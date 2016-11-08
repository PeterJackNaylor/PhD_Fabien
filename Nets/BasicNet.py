import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import sys
import os


# Different types of layers and fillers.

# Fillers:


Gaussian_fil = dict(type="gaussian", std=0.01)
Constant_fil = dict(type="constant", value=0)
param1_1 = dict(lr_mult=1, decay_mult=1)
param2_0 = dict(lr_mult=2, decay_mult=0)
noth = dict(lr_mult=0)
xavier = dict(type="xavier")

# Basic layer


def Conv(bottom, nout, ks=3, stride=1, pad=1, weight_filler=xavier):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[param1_1, param2_0],
                         weight_filler=xavier)
    return conv


def Relu(bottom):
    return L.ReLU(bottom, in_place=True)


def Maxpool(bottom, ks=2, stride=2, ntop=None):
    if ntop is None:
        layer = L.Pooling(bottom, pool=P.Pooling.MAX,
                          kernel_size=ks, stride=stride)
    else:
        layer = L.Pooling(bottom, pool=P.Pooling.MAX,
                          kernel_size=ks, stride=stride, ntop=ntop)
    return layer


def Maxunpool(bottom1, bottom2, bottom3, ks=2, stride=2):
    unpooling_param = dict(pool=P.Pooling.MAX, kernel_size=ks,
                           stride=stride)
    return L.Unpooling(bottom1, bottom2, bottom3, pooling_param=unpooling_param)


def BatchNormalizer(bottom):
    param = [noth, noth, noth]
    bn = L.BatchNorm(bottom, param=param, in_place=True)
    sl = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True)
    return bn, sl


def Deconv(bottom, nout, pad, ks, stride, weight_filler, bias_filler):
    deconv = L.Deconvolution(bottom,
                             param=[param1_1, param2_0],
                             convolution_param=dict(num_output=nout,
                                                    pad=pad,
                                                    kernel_size=ks,
                                                    stride=stride,
                                                    weight_filler=weight_filler,
                                                    bias_filler=bias_filler))
    return deconv

# Combination of layers


def ConvRelu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[param1_1, param2_0],
                         weight_filler=xavier,
                         bias_filler=Constant_fil)
    return conv, L.ReLU(conv, in_place=True)


def ConvBnRelu(bottom, nout, ks=3, stride=1, pad=1):
    conv = Conv(bottom, nout, ks, stride, pad)
    bn, sl = BatchNormalizer(conv)
    relu = Relu(sl)
    return conv, bn, sl, relu


def DeconvBnRelu(bottom, nout, ks=3, pad=0, weight_filler=Gaussian_fil, bias_filler=Constant_fil):
    deconv = Deconv(bottom, nout, ks, pad, weight_filler, bias_filler)
    bn, sl = BatchNormalizer(deconv)
    relu = Relu(sl)
    return deconv, bn, sl, relu


def DeconvReConvReConvRe(bottom1, val, deconv_out=None):
    if deconv_out is None:
        deconv_out = val

    deconv = Deconv(bottom1, deconv_out, 2, 2,
                    xavier, Constant_fil)
    relu1 = Relu(deconv)
    conv2, relu2 = conv_relu(relu1, val)
    conv3, relu3 = conv_relu(conv2, val)

    return deconv, relu1, conv2, relu2, conv3, relu3


def DeconvReCropConcatConvReConvRe(bottom1, bridge2, val, deconv_out=None):
    if deconv_out is None:
        deconv_out = val

    deconv = Deconv(bottom1, deconv_out, 2, 2,
                    xavier, Constant_fil)
    relu1 = Relu(deconv)
    croped = crop(bridge2, relu1)
    concat = L.Concat(relu1, croped)
    conv2, relu2 = conv_relu(concat, val)
    conv3, relu3 = conv_relu(conv2, val)

    return deconv, relu1, croped, concat, conv2, relu2, conv3, relu3


def DataLayer(split, data_gene, batch_size, cn, Weight):

    pydata_params = dict()
    pydata_params["split"] = slit
    pydata_params["batch_size"] = batch_size
    pydata_params["classifier_name"] = cn
    pydata_params["datagen"] = data_gene
    pydata_params["mean"] = (104.00699, 116.66877, 122.67892)
    pydata_params["seed"] = 1337
    pydata_params["Weight"] = Weight

    pylayer = 'DataLayer'

    if not Weight:
        data, label = L.Python(module='CustomLayersPeter', layer=pylayer,
                               ntop=2, param_str=str(pydata_params))
        return data, label
    else:
        data, label, weight = L.Python(module='DataLayerPeter', layer=pylayer,
                                       ntop=3, param_str=str(pydata_params))
        return data, label, weight


def LossLayer(score, label, loss, weight=None):
    if loss == "softmax":
        losslayer = L.SoftmaxWithLoss(score, label,
                                      loss_param=dict(normalize=False))
    elif loss == "weight":
        losslayer = L.Python(score, label, weight,
                             module='DataLayerPeter', layer="WeigthedLossLayer")
    elif loss == "weightcpp":
        losslayer = L.WeightedSoftmaxLoss(score, label, weight,
                                          loss_param=dict(normalize=False))
    return losslayer
