import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import sys
import os


sys.path.append(os.getcwd())


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def fcn(split, data_path, classifier_name="FCN32",
        classifier_name1="score_fr", classifier_name2="upscore"):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
                         seed=1337, classifier_name=classifier_name)
#    pydata_params['dir'] = data_path
    pylayer = 'DataLayerPeter'
    pydata_params["datagen"] = data_path
    if split != "val":
        n.data, n.label = L.Python(module='DataLayerPeter', layer=pylayer,
                                   ntop=2, param_str=str(pydata_params))
    else:
        n.data = L.Data(input_param=dict(shape=dict(dim=[1, 3, 512, 512])))
        # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    score_fr = L.Convolution(n.drop7, num_output=2, kernel_size=1, pad=0,
                             param=[dict(lr_mult=1, decay_mult=1),
                                    dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type="xavier"))

    n.__setattr__(classifier_name1, score_fr)

    upscore = L.Deconvolution(score_fr,
                              convolution_param=dict(num_output=2, kernel_size=64, stride=32,
                                                     bias_term=False),
                              weight_filler=dict(type='bilinear'),
                              param=[dict(lr_mult=1)])
    n.__setattr__(classifier_name2, upscore)

    n.score = crop(upscore, n.data)

    if split != "val":
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
                                   loss_param=dict(normalize=False))  # , ignore_label=255))
        #n.acc = L.Accuracy(n.score2, n.label)
    return n.to_proto()


def make_net(wd, data_path, classifier_name="FCN32",
             classifier_name1="score_fr", classifier_name2="upscore"):
    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(fcn('train', data_path, classifier_name,
                        classifier_name1, classifier_name2)))

    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(fcn('test', data_path, classifier_name,
                        classifier_name1, classifier_name2)))

    with open(os.path.join(wd, 'deploy.prototxt'), 'w') as f:
        f.write(str(fcn('val', data_path, classifier_name,
                        classifier_name1, classifier_name2)))
