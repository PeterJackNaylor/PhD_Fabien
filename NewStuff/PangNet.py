# -*- coding: utf-8 -*-

import tensorflow as tf
import BasicNetTF as PTF
import numpy as np
from DataGen2 import DataGen, ListTransform
import pdb
import os
import time



CUDA_NODE = 0
SAVE_DIR = "/tmp/Pang/4"
N_ITER_MAX = 2000
N_TRAIN_SAVE = 4
N_TEST_SAVE = 8
LEARNING_RATE = 0.01
MOMENTUM = 0.99
MEAN = np.array([122.67892, 116.66877 ,104.00699])
HEIGHT = 224 
WIDTH = 224
CROP = 4
PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
BATCH_SIZE = 8
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
            key = [0,0,0,0]
            ImagesBatch[i], LabelsBatch[i,:,:,0] = DGTest[key]   
            ImagesBatch[i] -= Mean

    return {Inputs: ImagesBatch, Labels: LabelsBatch, PhaseTrain: Train}

def PangModel(Input, PhaseTrain, ImageSizeIn, ImageSizeOut, KeepProbability):

    ks = 5
    padding = "SAME"

    Out1_1 = PTF.ConvBatchLayer(Input, 3, 8, ks, "conv1_1", padding, PhaseTrain, S)
    Out1_2 = PTF.ConvBatchLayer(Out1_1, 8, 8, ks, "conv1_2", padding, PhaseTrain, S)
    Out1_3 = PTF.ConvBatchLayer(Out1_2, 8, 8, ks, "conv1_3", padding, PhaseTrain, S)


    logits = PTF.ConvLayer(Out1_3, 8, 2, ks, "logits", padding, PhaseTrain, S)
    annotation_pred = tf.argmax(logits, dimension=3, name="Prediction")
    print logits.get_shape()
    return tf.expand_dims(annotation_pred, dim=3), logits

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
    optimizer = (tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(LossVal, global_step=global_step))
#    return optimizer.apply_gradients(grads)
    return optimizer

def main(argv=None):

    weight_file = WEIGHT_FILE
    KeepProbability = tf.placeholder(tf.float32, name="keep_probabilty")
    PhaseTrain = tf.placeholder(tf.bool, name='phase_train')
    Input, Label = PTF.InputLayer(WIDTH, WIDTH, 3, PhaseTrain, S, BS=BATCH_SIZE, Weight=False)
    

    LabelInt = tf.to_int64(Label)
    Predicted, Logits = PangModel(Input, PhaseTrain, WIDTH, WIDTH, KeepProbability)

    if S:
        tf.summary.image("pred_annotation", tf.cast(Predicted, tf.uint8), max_outputs=4)

    with tf.name_scope('Loss'):
        onehot_labels = tf.one_hot(indices=tf.cast(tf.squeeze(Label, axis=3), tf.int32), depth=2)
        print onehot_labels.get_shape()
        Loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=Logits)
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

    if ckpt and ckpt.model_checkpoint_path:

        saver.restore(sess, ckpt.model_checkpoint_path)
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