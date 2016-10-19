import caffe
from caffe import layers as L, params as P
import os
Constant_fil = dict(type="constant", value=0)


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1),
                                dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type="xavier"),
                         bias_filler=Constant_fil)
    return conv, L.ReLU(conv, in_place=True)


def BaochuanNet(split, data_gene, batch_size=1, classifier_name="BaochuanNet"):
    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
                         seed=1337, batch_size=batch_size, classifier_name=classifier_name)
#    pydata_params['dir'] = data_path
    pylayer = 'DataLayerPeter'
    pydata_params["datagen"] = data_gene

    n.data, n.label = L.Python(module='DataLayerPeter', layer=pylayer,
                               ntop=2, param_str=str(pydata_params))
    n.conv1, n.relu1 = conv_relu(n.data, 8, pad=1)
    n.conv2, n.relu2 = conv_relu(n.relu1, 8, pad=1)
    n.conv3, n.relu3 = conv_relu(n.relu2, 8, pad=1)

    score_fr = L.Convolution(n.relu3, num_output=2, kernel_size=1, pad=0,
                             param=[dict(lr_mult=1, decay_mult=1),
                                    dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type="xavier"))

    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=True, ignore_label=255))


def make_net(wd, data_gene_train, data_gene_test, batch_size=1, classifier_name="BaochuanNet"):
    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(BaochuanNet('train', data_gene_train, batch_size,
                                classifier_name)))
    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(BaochuanNet('test', data_gene_test,
                                1, classifier_name)))
