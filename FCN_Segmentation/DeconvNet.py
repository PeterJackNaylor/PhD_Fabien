# Change caffe version for the deconvnet

import sys

sys.path.append(
    "/data/users/pnaylor/Documents/Python/PhD_Fabien/UsefulFunctions/")
sys.path[4] = "/data/users/pnaylor/Documents/Python/caffe_unpool/caffe/python"


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

Gaussian_fil = dict(type="gaussian", std=0.01)
Constant_fil = dict(type="constant", value=0)


def Conv(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv


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


def BatchNormalizer(bottom):
    noth = dict(lr_mult=0)
    param = [noth, noth, noth]
    bn = L.BatchNorm(bottom, param=param)
    return bn


def Relu(bottom):
    return L.ReLU(bottom, in_place=True)


def ConvBnRelu(bottom, nout, ks=3, stride=1, pad=1):
    conv = Conv(bottom, nout, ks, stride, pad)
    bn = BatchNormalizer(conv)
    relu = Relu(bn)
    return conv, bn, relu


def DeconvBnRelu(bottom, nout, ks=3, pad=0, weight_filler=Gaussian_fil, bias_filler=Constant_fil):
    deconv = Deconv(bottom, nout, ks, pad, weight_filler, bias_filler)
    bn = BatchNormalizer(deconv)
    relu = Relu(bn)
    return deconv, bn, relu


def max_pool(bottom, ks=2, stride=2):
    layer = L.Pooling(bottom, pool=P.Pooling.MAX,
                      kernel_size=ks, stride=stride, ntop=3)
    return layer


def max_unpool(bottom1, bottom2, bottom3, unpool_size=14, ks=2, stride=2):
    unpooling_param = dict(pool=P.Pooling.MAX, kernel_size=ks,
                           stride=stride)
    # should be unpooling?
    return L.Unpooling(bottom1, bottom2, bottom3, pooling_param=unpooling_param)


def DeconvNet(split, data_gene, classifier_name="DeconvNet"):
    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
                         seed=1337, classifier_name=classifier_name)
    pylayer = 'DataLayerPeter'
    pydata_params["datagen"] = data_gene
    n.data, n.label = L.Python(module='DataLayerPeter', layer=pylayer,
                               ntop=2, param_str=str(pydata_params))
    n.conv1_1, n.bn1_1, n.relu1_1 = ConvBnRelu(n.data, 64, 3, 1, 1)
    n.conv1_2, n.bn1_2, n.relu1_2 = ConvBnRelu(n.relu1_1, 64, 3, 1, 1)

    n.pool1, n.pool1_mask = max_pool(n.relu1_2, ks=2, stride=2)

    n.conv2_1, n.bn2_1, n.relu2_1 = ConvBnRelu(n.pool1, 128, 3, 1, 1)
    n.conv2_2, n.bn2_2, n.relu2_2 = ConvBnRelu(n.relu2_1, 128, 3, 1, 1)

    n.pool2, n.pool2_mask, n.pool2_count = max_pool(n.relu2_2, ks=2, stride=2)

    n.conv3_1, n.bn3_1, n.relu3_1 = ConvBnRelu(n.pool2, 256, 3, 1, 1)
    n.conv3_2, n.bn3_2, n.relu3_2 = ConvBnRelu(n.relu3_1, 256, 3, 1, 1)
    n.conv3_3, n.bn3_3, n.relu3_3 = ConvBnRelu(n.relu3_2, 256, 3, 1, 1)

    pdb.set_trace()
    n.pool3, n.pool3_mask, n.pool3_count = max_pool(n.relu3_3, ks=2, stride=2)

    n.conv4_1, n.bn4_1, n.relu4_1 = ConvBnRelu(n.pool3, 512, 3, 1, 1)
    n.conv4_2, n.bn4_2, n.relu4_2 = ConvBnRelu(n.relu4_1, 512, 3, 1, 1)
    n.conv4_3, n.bn4_3, n.relu4_3 = ConvBnRelu(n.relu4_2, 512, 3, 1, 1)

    n.pool4, n.pool4_mask, n.pool4_count = max_pool(n.relu4_3, ks=2, stride=2)

    n.conv5_1, n.bn5_1, n.relu5_1 = ConvBnRelu(n.pool4, 512, 3, 1, 1)
    n.conv5_2, n.bn5_2, n.relu5_2 = ConvBnRelu(n.relu5_1, 512, 3, 1, 1)
    n.conv5_3, n.bn5_3, n.relu5_3 = ConvBnRelu(n.relu5_2, 512, 3, 1, 1)

    n.pool5, n.pool5_mask, n.pool5_count = max_pool(n.relu5_3, ks=2, stride=2)

    n.fc6, n.bn6, n.relu6 = ConvBnRelu(n.pool5, 4096, ks=7, stride=1, pad=0)
    #n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7, n.bn7, n.relu7 = ConvBnRelu(n.relu6, 4096, ks=1, stride=1, pad=0)
    #n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # no need for dropout as we have batch normalization!
    n.deconv_fc6, n.deconv_fc6_bn, n.deconv_fc6_relu = DeconvBnRelu(
        n.relu7, 512, ks=7)

    n.unpool5 = max_unpool(n.deconv_fc6_relu, n.pool5_mask,
                           n.pool5_count, unpool_size=14)

    n.deconv5_1, n.debn5_1, n.derelu5_1 = DeconvBnRelu(
        n.unpool5, 512, ks=3, pad=1)
    n.deconv5_2, n.debn5_2, n.derelu5_2 = DeconvBnRelu(
        n.derelu5_1, 512, ks=3, pad=1)
    n.deconv5_3, n.debn5_3, n.derelu5_3 = DeconvBnRelu(
        n.derelu5_2, 512, ks=3, pad=1)

    n.unpool4 = max_unpool(n.derelu5_3, n.pool4_mask,
                           n.pool4_count, unpool_size=28)

    n.deconv4_1, n.debn4_1, n.derelu4_1 = DeconvBnRelu(
        n.unpool4, 512, ks=3, pad=1)
    n.deconv4_2, n.debn4_2, n.derelu4_2 = DeconvBnRelu(
        n.derelu4_1, 512, ks=3, pad=1)
    n.deconv4_3, n.debn4_3, n.derelu4_3 = DeconvBnRelu(
        n.derelu4_2, 256, ks=3, pad=1)

    n.unpool3 = max_unpool(n.derelu4_3, n.pool3_mask,
                           n.pool3_count, unpool_size=56)

    n.deconv3_1, n.debn3_1, n.derelu3_1 = DeconvBnRelu(
        n.unpool3, 256, ks=3, pad=1)
    n.deconv3_2, n.debn3_2, n.derelu3_2 = DeconvBnRelu(
        n.derelu3_1, 256, ks=3, pad=1)
    n.deconv3_3, n.debn3_3, n.derelu3_3 = DeconvBnRelu(
        n.derelu3_2, 128, ks=3, pad=1)

    n.unpool2 = max_unpool(n.derelu3_3, n.pool2_mask,
                           n.pool2_count, unpool_size=112)

    n.deconv2_1, n.debn2_1, n.derelu2_1 = DeconvBnRelu(
        n.unpool2, 128, ks=3, pad=1)
    n.deconv2_2, n.debn2_2, n.derelu2_2 = DeconvBnRelu(
        n.derelu2_1, 64, ks=3, pad=1)

    n.unpool1 = max_unpool(n.derelu2_2, n.pool1_mask,
                           n.pool1_count, unpool_size=224)

    n.deconv1_1, n.debn1_1, n.derelu1_1 = DeconvBnRelu(
        n.unpool1, 64, ks=3, pad=1)
    n.deconv1_2, n.debn1_2, n.derelu1_2 = DeconvBnRelu(
        n.derelu1_1, 64, ks=3, pad=1)

    n.score = L.Convolution(n.derelu1_2, kernel_size=3, stride=1,
                            num_output=2, pad=1,
                            param=[dict(lr_mult=1, decay_mult=1), dict(
                                lr_mult=2, decay_mult=0)],
                            weight_filler=Gaussian_fil,
                            bias_filler=Constant_fil)
    if split != "val":
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
                                   loss_param=dict(normalize=False, ignore_label=255))
    return n.to_proto()


def make_net(wd, data_gene_train, data_gene_test, classifier_name="DeconvNet"):
    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(DeconvNet('train', data_gene_train, classifier_name)))
    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(DeconvNet('test', data_gene_test, classifier_name)))
    # with open(os.path.join(wd, 'deploy.prototxt'), 'w') as f:
    #    f.write(str(DeconvNet('val', data_gene_train, classifier_name)))
