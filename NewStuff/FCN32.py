# -*- coding: utf-8 -*-

import tensorflow as tf
import BasicNetTF as PTF
import numpy as np
from DataGen2 import DataGen, ListTransform
import pdb
import os
import time
from math import ceil



CUDA_NODE = 0
SAVE_DIR = "/tmp/FCN/1"
N_ITER_MAX = 2000
N_TRAIN_SAVE = 4
N_TEST_SAVE = 8
LEARNING_RATE = 1.
MEAN = np.array([122.67892, 116.66877 ,104.00699])
HEIGHT = 224 
WIDTH = 224
CROP = 4
PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
BATCH_SIZE = 1
LRSTEP = 200
SUMMARY = True
S = SUMMARY
WEIGHT_FILE = "/home/pnaylor/Downloads/vgg16_weights.npz"




os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)


transform_list, transform_list_test = ListTransform()
DG_TRAIN = DataGen(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                   transforms=transform_list)
DG_TEST  = DataGen(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                   transforms=transform_list_test)


def FeedDict(Train, Inputs, Labels, PhaseTrain, DGTrain=DG_TRAIN, DGTest=DG_TEST, 
              BatchSize=BATCH_SIZE, Width=WIDTH, Height=HEIGHT,
              Mean=MEAN, Dim=3):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

    ImagesBatch = np.zeros(shape=(BatchSize, Width, Height, Dim), dtype='float')
    LabelsBatch = np.zeros(shape=(BatchSize, Width, Height, 1), dtype='float')   

    if Train:

        for i in range(BatchSize):

            key = DGTrain.NextKeyRandList(0)
            ImagesBatch[i], LabelsBatch[i,:,:,0] = DGTrain[key]
            ImagesBatch[i] -= Mean

    else:

        for i in range(BatchSize):

            key = DGTest.NextKeyRandList(0)
            ImagesBatch[i], LabelsBatch[i,:,:,0] = DGTest[key]   
            ImagesBatch[i] -= Mean

    return {Inputs: ImagesBatch, Labels: LabelsBatch, PhaseTrain: Train}


def _upscore_layer(bottom, shape,
                   num_classes, name,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
                # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
#        output_shape = tf.pack(new_shape)
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, num_classes, in_features]

        # create
#        num_input = ksize * ksize * in_features / stride
#        stddev = (2 / num_input)**0.5

        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

#        if debug:
#            deconv = tf.Print(deconv, [tf.shape(deconv)],
#                              message='Shape of %s' % name,
#                              summarize=4, first_n=1)

    return deconv


def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
           shape=weights.shape)

def FCN32Model(Input, PhaseTrain, ImageSizeIn, ImageSizeOut, KeepProbability):

    ks = 3
    padding = "SAME"

    Out1_1, p1_1 = PTF.ConvLayer(Input, 3, 64, 3, "conv1_1", padding, PhaseTrain, S, fine_tune=True)
    Out1_2, p1_2 = PTF.ConvLayer(Out1_1, 64, 64, ks, "conv1_2", padding, PhaseTrain, S, fine_tune=True)

    MaxPool1 = tf.nn.max_pool(Out1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name="pool1")

    Out2_1, p2_1 = PTF.ConvLayer(MaxPool1, 64, 128, ks, "conv2_1", padding, PhaseTrain, S, fine_tune=True)
    Out2_2, p2_2 = PTF.ConvLayer(Out2_1, 128, 128, ks, "conv2_2", padding, PhaseTrain, S, fine_tune=True)
    
    MaxPool2 = tf.nn.max_pool(Out2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name="pool2")

    Out3_1, p3_1 = PTF.ConvLayer(MaxPool2, 128, 256, ks, "conv3_1", padding, PhaseTrain, S, fine_tune=True)
    Out3_2, p3_2 = PTF.ConvLayer(Out3_1, 256, 256, ks, "conv3_2", padding, PhaseTrain, S, fine_tune=True)
    Out3_3, p3_3 = PTF.ConvLayer(Out3_2, 256, 256, ks, "conv3_3", padding, PhaseTrain, S, fine_tune=True)

    MaxPool3 = tf.nn.max_pool(Out3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name="pool3")

    Out4_1, p4_1 = PTF.ConvLayer(MaxPool3, 256, 512, ks, "conv4_1", padding, PhaseTrain, S, fine_tune=True)
    Out4_2, p4_2 = PTF.ConvLayer(Out4_1, 512, 512, ks, "conv4_2", padding, PhaseTrain, S, fine_tune=True)
    Out4_3, p4_3 = PTF.ConvLayer(Out4_2, 512, 512, ks, "conv4_2", padding, PhaseTrain, S, fine_tune=True)

    MaxPool4 = tf.nn.max_pool(Out4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name="pool4")

    Out5_1, p5_1 = PTF.ConvLayer(MaxPool4, 512, 512, ks, "conv5_1", padding, PhaseTrain, S, fine_tune=True)
    Out5_2, p5_2 = PTF.ConvLayer(Out5_1, 512, 512, ks, "conv5_2", padding, PhaseTrain, S, fine_tune=True)
    Out5_3, p5_3 = PTF.ConvLayer(Out5_2, 512, 512, ks, "conv5_3", padding, PhaseTrain, S, fine_tune=True)

    MaxPool5 = tf.nn.max_pool(Out5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name="pool5")
    print "Pool", MaxPool5.get_shape()
    fc6, p6 = PTF.ConvLayer(MaxPool5, 512, 4096, 7, "fc6", padding, PhaseTrain, S, fine_tune=True)
    fc7, p7 = PTF.ConvLayer(fc6, 4096, 4096, 1, "fc7", padding, PhaseTrain, S, fine_tune=True)

    score_fr_32, p32 = PTF.ConvLayer(fc7, 4096, 2, 1, "score_fr_32", padding, PhaseTrain, S, fine_tune=True)

    logits = _upscore_layer(score_fr_32, shape=tf.shape(Input),
                             num_classes=2, name='up', 
                             ksize=64, stride=32)

    annotation_pred = tf.argmax(logits, dimension=3, name="Prediction")
    variables = p1_1 + p1_2 + p2_1 + p2_2 + p3_1 + p3_2 + p3_3 + \
                p4_1 + p4_2 + p4_3 + p5_1 + p5_2 + p5_3 + p6 + p7

    return tf.expand_dims(annotation_pred, dim=3), logits, variables

def load_weights(weight_file, variables, sess):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        print i, k, np.shape(weights[k])
        sess.run(variables[i].assign(weights[k]))


def train(LossVal, VarList, LearningRate, LRUpdate):

    k = 0.96
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate =  LearningRate
    #pdb.set_trace()
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                LRUpdate, k, staircase=True)
    tf.summary.scalar('LearningRate', learning_rate)

#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#    grads = optimizer.compute_gradients(LossVal, var_list=VarList)
    optimizer = (tf.train.GradientDescentOptimizer(learning_rate).minimize(LossVal, global_step=global_step))

#    return optimizer.apply_gradients(grads)
    return optimizer

def main(argv=None):

    weight_file = WEIGHT_FILE
    KeepProbability = tf.placeholder(tf.float32, name="keep_probabilty")
    PhaseTrain = tf.placeholder(tf.bool, name='phase_train')
    Input, Label = PTF.InputLayer(WIDTH, WIDTH, 3, PhaseTrain, S, BS=BATCH_SIZE, Weight=False)
    

    LabelInt = tf.to_int64(Label)
    Predicted, Logits, variables = FCN32Model(Input, PhaseTrain, WIDTH, WIDTH, KeepProbability)

    if S:
        tf.summary.image("pred_annotation", tf.cast(Predicted, tf.uint8), max_outputs=4)

    with tf.name_scope('Loss'):

        Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Logits,
                                                                              labels=tf.squeeze(LabelInt, squeeze_dims=[3]),
                                                                              name="entropy"))
        tf.summary.scalar('Loss', Loss)

    with tf.name_scope('Accuracy'):
 
        CorrectPrediction = tf.equal(Predicted, LabelInt)
        Accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))

        with tf.name_scope('F1'):

            with tf.name_scope('Subs'):

                TP = tf.count_nonzero(Predicted * LabelInt)
                TN = tf.count_nonzero((Predicted - 1) * (LabelInt - 1))
                FP = tf.count_nonzero(Predicted * (LabelInt - 1))
                FN = tf.count_nonzero((Predicted - 1) * LabelInt)

            with tf.name_scope('Precision'):

                precision = tf.divide(TP, tf.add(TP, FP))

            with tf.name_scope('Recall'):

                recall = tf.divide(TP, tf.add(TP, TN))

            with tf.name_scope('F1'):

                num = tf.multiply(precision, recall)
                dem = tf.add(precision, recall)
                F1 = tf.scalar_mul(2, tf.divide(num, dem))

        tf.summary.scalar('Accuracy', Accuracy)
        tf.summary.scalar('Precision', precision)
        tf.summary.scalar('Recall', recall)
        tf.summary.scalar('F1', F1)


    trainable_var = tf.trainable_variables()
    train_op = train(Loss, trainable_var, LEARNING_RATE, LRSTEP)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()

    print("Setting up Saver...")

    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SAVE_DIR + '/test')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(SAVE_DIR + '/train')

#    if ckpt and ckpt.model_checkpoint_path:
    if True:
#        saver.restore(sess, ckpt.model_checkpoint_path)
        load_weights(weight_file, variables, sess)
        print("Model restored...")
    TIME = time.time()
    for steps in range(1, N_ITER_MAX+1):

        feed_dict_train = FeedDict(True, Input, Label, PhaseTrain)
        sess.run(train_op, feed_dict=feed_dict_train)

        if steps % N_TRAIN_SAVE == 0:
            train_loss, train_acc, train_f1, summary_str = sess.run([Loss, Accuracy, F1, merged_summary], feed_dict=feed_dict_train)
            train_writer.add_summary(summary_str, steps)
            print('Training loss {:5d}: {:.2f}, Accuracy: {:.2f}, F1: {:.2f}'.format(steps, train_loss, train_acc, train_f1))
            diff_time = time.time() - TIME
            TIME = time.time()
            print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
        if steps % N_TEST_SAVE == 0:
            feed_dict_test = FeedDict(False, Input, Label, PhaseTrain)
            test_loss, test_acc, test_f1 = sess.run([Loss, Accuracy, F1], feed_dict=feed_dict_test)
            test_writer.add_summary(summary_str, steps)
            print('Validation loss {:5d}: {:.2f}, Accuracy: {:.2f}, F1: {:.2f}'.format(steps, test_loss, test_acc, test_f1))
            saver.save(sess, SAVE_DIR + '/test/' + "model.ckpt", steps)



if __name__ == "__main__":
    tf.app.run()