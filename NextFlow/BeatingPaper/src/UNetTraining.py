# -*- coding: utf-8 -*-
from optparse import OptionParser
from Nets.UNetBatchNorm_v2 import UNetBatchNorm
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from Data.DataGenClass import DataGenMulti
from Data.DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import math
import pdb

class unet_diff(UNetBatchNorm):
    def init_training_graph(self):

        with tf.name_scope('Evaluation'):
            self.logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            self.predictions = tf.argmax(self.logits, axis=3)

            with tf.name_scope('Loss'):
                condition = tf.equal(self.train_labels_node, 255)
                case_true = tf.ones(self.train_labels_node.get_shape())
                case_false = self.train_labels_node
                anno = tf.where(condition, case_true, case_false)
                anno = tf.squeeze(tf.cast(anno, tf.int32), squeeze_dims=[3])
#                anno = tf.divide(anno, 255.)
                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=anno,
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)
            with tf.name_scope('Accuracy'):

                LabelInt = tf.squeeze(tf.cast(self.train_labels_node, tf.int64), squeeze_dims=[3])
                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('Prediction'):

                self.TP = tf.count_nonzero(self.predictions * LabelInt)
                self.TN = tf.count_nonzero((self.predictions - 1) * (LabelInt - 1))
                self.FP = tf.count_nonzero(self.predictions * (LabelInt - 1))
                self.FN = tf.count_nonzero((self.predictions - 1) * LabelInt)

            with tf.name_scope('Precision'):

                self.precision = tf.divide(self.TP, tf.add(self.TP, self.FP))
                tf.summary.scalar('Precision', self.precision)

            with tf.name_scope('Recall'):

                self.recall = tf.divide(self.TP, tf.add(self.TP, self.FN))
                tf.summary.scalar('Recall', self.recall)
            with tf.name_scope('F1'):

                num = tf.multiply(self.precision, self.recall)
                dem = tf.add(self.precision, self.recall)
                self.F1 = tf.scalar_mul(2, tf.divide(num, dem))
                tf.summary.scalar('F1', self.F1)

            with tf.name_scope('MeanAccuracy'):

                Nprecision = tf.divide(self.TN, tf.add(self.TN, self.FN))
                self.MeanAcc = tf.divide(tf.add(self.precision, Nprecision) ,2)
                tf.summary.scalar('Performance', self.MeanAcc)
            #self.batch = tf.Variable(0, name = "batch_iterator")

            self.train_prediction = tf.nn.softmax(self.logits)

            self.test_prediction = tf.nn.softmax(self.logits)

        tf.global_variables_initializer().run()


        print('Computational graph initialised')


if __name__== "__main__":

    parser = OptionParser()

#    parser.add_option("--gpu", dest="gpu", default="0", type="string",
#                      help="Input file (raw data)")
    parser.add_option("--tf_record", dest="TFRecord", type="string",
                      help="Where to find the TFrecord file")

    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")

    parser.add_option("--log", dest="log",
                      help="log dir")

    parser.add_option("--learning_rate", dest="lr", type="float",
                      help="learning_rate")

    parser.add_option("--batch_size", dest="bs", type="int",
                      help="batch size")

    parser.add_option("--epoch", dest="epoch", type="int",
                      help="number of epochs")

    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")

    parser.add_option("--weight_decay", dest="weight_decay", type="float",
                      help="weight decay value")

    parser.add_option("--dropout", dest="dropout", type="float",
                      default=0.5, help="dropout value to apply to the FC layers.")

    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")

    parser.add_option('--n_threads', dest="THREADS", type=int, default=100,
                      help="number of threads to use for the preprocessing.")

    parser.add_option('--crop', dest="crop", type=int, default=4,
                      help="crop size depending on validation/test/train phase.")

    (options, args) = parser.parse_args()

#    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

    TFRecord = options.TFRecord
    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    DROPOUT = options.dropout
    MEAN_FILE = options.mean_file 
    N_THREADS = options.THREADS



    LEARNING_RATE = options.lr
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str

    
    
    HEIGHT = 224 
    WIDTH = 224
    
    
    BATCH_SIZE = options.bs
    LRSTEP = "10epoch"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch

    PATH = options.path


    HEIGHT = 212
    WIDTH = 212
    SIZE = (HEIGHT, WIDTH)

    N_TRAIN_SAVE = 100
 
    CROP = options.crop


    transform_list, transform_list_test = ListTransform()

    TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach", "validation"]
    DG_TRAIN = DataGenMulti(PATH, split='train', crop = 16, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file=None, num=TEST_PATIENT)

    DG_TRAIN.SetPatient(TEST_PATIENT)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE
    
    TEST_PATIENT = ["test"]

    DG_TEST  = DataGenMulti(PATH, split="test", crop = 4, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE, num=TEST_PATIENT)
    DG_TEST.SetPatient(TEST_PATIENT)

    model = unet_diff(TFRecord,    LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=DROPOUT)

    model.train(DG_TEST)
