import tensorflow as tf
import numpy as np
DEBUG=False

XAVIER = tf.contrib.layers.xavier_initializer_conv2d()
CONSTANT_INIT = 0.1




def InputLayer(ImageSizeIn, ImageSizeOut, Dim, PhaseTrain, Summary = True, BS=None, Weight=False):
    with tf.name_scope('input'):
        # None -> batch size can be any size
        x_ = tf.placeholder(tf.float32, shape=[BS, ImageSizeIn, ImageSizeIn, Dim], name="image-input") 
        y_ = tf.placeholder(tf.float32, shape=[BS, ImageSizeOut, ImageSizeOut, 1], name="label-input")

        if Summary:
            tf.summary.image("Input" ,x_, max_outputs=4)
            tf.summary.image("Label", y_, max_outputs=4)
        
        if Weight:
            w_ = tf.placeholder(tf.float32, shape=[BS, ImageSizeOut, ImageSizeOut, 1], name="weight-input")
            return x_, y_, w_

        else:
            return x_, y_

def print_dim(text ,tensor):
    print text, tensor.get_shape()
    print 

def ConvLayer(Input, ChannelsIn, ChannelsOut, ks, Name, padding, PhaseTrain, Summary = True, debug=DEBUG):
    with tf.name_scope(Name):

        StddevW = 1 / float(ks ** 2 * (ChannelsOut + ChannelsIn))
        ShapeW = [ks,ks, ChannelsIn, ChannelsOut]
        init = tf.truncated_normal(ShapeW, stddev=StddevW)
        w_ = tf.Variable(init, name="W")
        b_ = tf.Variable(tf.constant(CONSTANT_INIT, shape=[ChannelsOut]), name="B")
        conv = tf.nn.conv2d(Input, w_, strides=[1,1,1,1], padding=padding)
        act = tf.nn.relu(conv + b_)

        if Summary:
            tf.summary.histogram("weights", w_)
            tf.summary.histogram("biases", b_)
            tf.summary.histogram("activations", act)
        if debug:
            print_dim(Name, act)
        return act

def BatchNorm(Input, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.name_scope(scope):
        init_beta = tf.constant(0.0, shape=[n_out])
        beta = tf.Variable(init_beta, name="beta")
        init_gamma = tf.random_normal([n_out], 1.0, 0.02)
        gamma = tf.Variable(init_gamma)
        batch_mean, batch_var = tf.nn.moments(Input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(Input, mean, var, beta, gamma, eps)
    return normed

def ConvBatchLayer(Input, ChannelsIn, ChannelsOut, ks, Name, padding, PhaseTrain, Summary = True, debug=DEBUG):
    with tf.name_scope(Name):

        StddevW = 1 / float(ks ** 2 * (ChannelsOut + ChannelsIn))
        ShapeW = [ks,ks, ChannelsIn, ChannelsOut]
        init = tf.truncated_normal(ShapeW, stddev=StddevW)
        w_ = tf.Variable(init, name="W")
        b_ = tf.Variable(tf.constant(CONSTANT_INIT, shape=[ChannelsOut]), name="B")
        conv = tf.nn.conv2d(Input, w_, strides=[1,1,1,1], padding=padding)
        conv_bn = BatchNorm(conv, ChannelsOut, PhaseTrain, scope=Name + "bn")
        act = tf.nn.relu(conv_bn + b_)

        if Summary:
            tf.summary.histogram("weights", w_)
            tf.summary.histogram("biases", b_)
            tf.summary.histogram("activations", act)
        if debug:
            print_dim(Name, act)
        return act


def ConvBatchLayerWithoutRelu(Input, ChannelsIn, ChannelsOut, ks, Name, padding, PhaseTrain, Summary = True, debug=DEBUG):
    with tf.name_scope(Name):

        StddevW = 1 / float(ks ** 2 * (ChannelsOut + ChannelsIn))
        ShapeW = [ks,ks, ChannelsIn, ChannelsOut]
        init = tf.truncated_normal(ShapeW, stddev=StddevW)
        w_ = tf.Variable(init, name="W")
        conv = tf.nn.conv2d(Input, w_, strides=[1,1,1,1], padding=padding)
        conv_bn = BatchNorm(conv, ChannelsOut, PhaseTrain,  scope=Name + "bn")
        if Summary:
            tf.summary.histogram("weights", w_)
        if debug:
            print_dim(Name, conv)

        return conv_bn

def ConvLayerWithoutRelu(Input, ChannelsIn, ChannelsOut, ks, Name, padding, PhaseTrain, Summary = True, debug=DEBUG):
    with tf.name_scope(Name):

        StddevW = 1 / float(ks ** 2 * (ChannelsOut + ChannelsIn))
        ShapeW = [ks,ks, ChannelsIn, ChannelsOut]
        init = tf.truncated_normal(ShapeW, stddev=StddevW)
        w_ = tf.Variable(init, name="W")
        conv = tf.nn.conv2d(Input, w_, strides=[1,1,1,1], padding=padding)
        if Summary:
            tf.summary.histogram("weights", w_)
        if debug:
            print_dim(Name, conv)

        return conv


def TransposeConvBatchLayer(Input, ChannelsIn, ChannelsOut, ks, Name, padding, PhaseTrain,
                            Summary = True, debug=DEBUG):
    """ This one doesn't seem to work... """
    with tf.name_scope(Name):
        StddevW = 1 / float(ks ** 2 * (ChannelsOut + ChannelsIn))
        ShapeW = [ks,ks, ChannelsOut, ChannelsIn]
        init = tf.truncated_normal(ShapeW, stddev=StddevW)
        w_ = tf.Variable(init, name="W")
        b_ = tf.Variable(tf.constant(CONSTANT_INIT, shape=[ChannelsOut]), name="B")
        InputShape = tf.shape(Input)
        OutputShape = tf.stack([InputShape[0], InputShape[1]*2, InputShape[2]*2, InputShape[3]//2])

        trans_conv = tf.nn.conv2d_transpose(Input, w_, output_shape=OutputShape,
                                            strides=[1,ks,ks,1], padding=padding)
        conv_bn = BatchNorm(trans_conv, ChannelsIn, PhaseTrain, scope=Name + "bn")
        act = tf.nn.relu(conv_bn + b_)

        if Summary:
            tf.summary.histogram("weights", w_)
            tf.summary.histogram("biases", b_)
            tf.summary.histogram("activations", act)

        if debug:
            print_dim(Name, act)

        return act

def TransposeConvLayer(Input, ChannelsIn, ChannelsOut, ks, Name, padding, PhaseTrain,
                            Summary = True, debug=DEBUG):
    with tf.name_scope(Name):
        StddevW = 1 / float(ks ** 2 * (ChannelsOut + ChannelsIn))
        ShapeW = [ks,ks, ChannelsOut, ChannelsIn]
        init = tf.truncated_normal(ShapeW, stddev=StddevW)
        w_ = tf.Variable(init, name="W")
        b_ = tf.Variable(tf.constant(CONSTANT_INIT, shape=[ChannelsOut]), name="B")
        InputShape = tf.shape(Input)
        OutputShape = tf.stack([InputShape[0], InputShape[1]*2, InputShape[2]*2, InputShape[3]//2])

        trans_conv = tf.nn.conv2d_transpose(Input, w_, output_shape=OutputShape,
                                            strides=[1,ks,ks,1], padding=padding)

        act = tf.nn.relu(trans_conv + b_)

        if Summary:
            tf.summary.histogram("weights", w_)
            tf.summary.histogram("biases", b_)
            tf.summary.histogram("activations", act)

        return act

import pdb

def CropAndMerge(Input1, Input2, Size1, Size2, Name, debug=DEBUG):
    ### Size1 > Size2
    with tf.name_scope(Name):
        assert Size1 > Size2 
        diff = (Size1 - Size2) / 2

        crop = tf.slice(Input1, [0, diff, diff, 0], [-1, Size2, Size2, -1])
        concat = tf.concat([crop, Input2], axis=3)
        if debug:
            print_dim(Name, Input1)
            print_dim(Name, Input2)
            print_dim("ConcatResult", concat)
        return concat

