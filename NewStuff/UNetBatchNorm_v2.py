# -*- coding: utf-8 -*-
from optparse import OptionParser
from UNetObject_v2 import UNet
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import math
import pdb

class UNetBatchNorm(UNet):

    def conv_layer_f(self, i_layer, w_var, scope_name, strides=[1,1,1,1], padding="VALID"):
        with tf.name_scope(scope_name):
            conv = tf.nn.conv2d(i_layer, w_var, strides=strides, padding=padding)
            n_out = w_var.shape[3].value
            BN = self.BatchNorm(conv, n_out, self.is_training)
            #BN = conv
            return BN


    def init_vars(self):

        #### had to add is_training, self.reuse

        self.is_training = tf.placeholder_with_default([True], shape=[1])
        ####

        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        n_features = self.N_FEATURES

        self.conv1_1weights = self.weight_xavier(3, self.NUM_CHANNELS, n_features, "conv1_1/")
        self.conv1_1biases = self.biases_const_f(0.1, n_features, "conv1_1/")

        self.conv1_2weights = self.weight_xavier(3, n_features, n_features, "conv1_2/")
        self.conv1_2biases = self.biases_const_f(0.1, n_features, "conv1_2/")

        self.conv1_3weights = self.weight_xavier(3, 2 * n_features, n_features, "conv1_3/")
        self.conv1_3biases = self.biases_const_f(0.1, n_features, "conv1_3/")

        self.conv1_4weights = self.weight_xavier(3, n_features, n_features, "conv1_4/")
        self.conv1_4biases = self.biases_const_f(0.1, n_features, "conv1_4/")



        self.conv2_1weights = self.weight_xavier(3, n_features, 2 * n_features, "conv2_1/")
        self.conv2_1biases = self.biases_const_f(0.1, 2 * n_features, "conv2_1/")

        self.conv2_2weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_2/")
        self.conv2_2biases = self.biases_const_f(0.1, 2 * n_features, "conv2_2/")

        self.conv2_3weights = self.weight_xavier(3, 4 * n_features, 2 * n_features, "conv2_3/")
        self.conv2_3biases = self.biases_const_f(0.1, 2 * n_features, "conv2_3/")

        self.conv2_4weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_4/")
        self.conv2_4biases = self.biases_const_f(0.1, 2 * n_features, "conv2_4/")



        self.conv3_1weights = self.weight_xavier(3, 2 * n_features, 4 * n_features, "conv3_1/")
        self.conv3_1biases = self.biases_const_f(0.1, 4 * n_features, "conv3_1/")

        self.conv3_2weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_2/")
        self.conv3_2biases = self.biases_const_f(0.1, 4 * n_features, "conv3_2/")

        self.conv3_3weights = self.weight_xavier(3, 8 * n_features, 4 * n_features, "conv3_3/")
        self.conv3_3biases = self.biases_const_f(0.1, 4 * n_features, "conv3_3/")

        self.conv3_4weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_4/")
        self.conv3_4biases = self.biases_const_f(0.1, 4 * n_features, "conv3_4/")



        self.conv4_1weights = self.weight_xavier(3, 4 * n_features, 8 * n_features, "conv4_1/")
        self.conv4_1biases = self.biases_const_f(0.1, 8 * n_features, "conv4_1/")

        self.conv4_2weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_2/")
        self.conv4_2biases = self.biases_const_f(0.1, 8 * n_features, "conv4_2/")

        self.conv4_3weights = self.weight_xavier(3, 16 * n_features, 8 * n_features, "conv4_3/")
        self.conv4_3biases = self.biases_const_f(0.1, 8 * n_features, "conv4_3/")

        self.conv4_4weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_4/")
        self.conv4_4biases = self.biases_const_f(0.1, 8 * n_features, "conv4_4/")



        self.conv5_1weights = self.weight_xavier(3, 8 * n_features, 16 * n_features, "conv5_1/")
        self.conv5_1biases = self.biases_const_f(0.1, 16 * n_features, "conv5_1/")

        self.conv5_2weights = self.weight_xavier(3, 16 * n_features, 16 * n_features, "conv5_2/")
        self.conv5_2biases = self.biases_const_f(0.1, 16 * n_features, "conv5_2/")




        self.tconv5_4weights = self.weight_xavier(2, 8 * n_features, 16 * n_features, "tconv5_4/")
        self.tconv5_4biases = self.biases_const_f(0.1, 8 * n_features, "tconv5_4/")

        self.tconv4_3weights = self.weight_xavier(2, 4 * n_features, 8 * n_features, "tconv4_3/")
        self.tconv4_3biases = self.biases_const_f(0.1, 4 * n_features, "tconv4_3/")

        self.tconv3_2weights = self.weight_xavier(2, 2 * n_features, 4 * n_features, "tconv3_2/")
        self.tconv3_2biases = self.biases_const_f(0.1, 2 * n_features, "tconv3_2/")

        self.tconv2_1weights = self.weight_xavier(2, n_features, 2 * n_features, "tconv2_1/")
        self.tconv2_1biases = self.biases_const_f(0.1, n_features, "tconv2_1/")



        self.logits_weight = self.weight_xavier(1, n_features, self.NUM_LABELS, "logits/")
        self.logits_biases = self.biases_const_f(0.1, self.NUM_LABELS, "logits/")

        self.keep_prob = tf.Variable(self.DROPOUT, name="dropout_prob")

        print('Model variables initialised')


    def Validation(self, DG_TEST, step):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.

            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval,
                             self.is_training: False}
                l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, s = self.sess.run([self.loss, 
                                                                                            self.accuracy, self.F1,
                                                                                            self.recall, self.precision,
                                                                                            self.MeanAcc,
                                                                                            self.merged_summary], feed_dict=feed_dict)
                l += l_tmp
                acc += acc_tmp
                F1 += F1_tmp
                recall += recall_tmp
                precision += precision_tmp
                meanacc += meanacc_tmp

            l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch

            summary = tf.Summary()
            summary.value.add(tag="TestMan/Accuracy", simple_value=acc)
            summary.value.add(tag="TestMan/Loss", simple_value=l)
            summary.value.add(tag="TestMan/F1", simple_value=F1)
            summary.value.add(tag="TestMan/Recall", simple_value=recall)
            summary.value.add(tag="TestMan/Precision", simple_value=precision)
            summary.value.add(tag="TestMan/Performance", simple_value=meanacc)
            self.summary_test_writer.add_summary(summary, step) 

            self.summary_test_writer.add_summary(s, step) 
            print('  Validation loss: %.1f' % l)
            print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
            self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)
        

    def train(self, DGTest):

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model()
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(steps):      
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                     (l, acc, acc1, recall, prec, f1))
                self.Validation(DG_TEST, step)
        coord.request_stop()
        coord.join(threads)


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

    N_TRAIN_SAVE = 500
 
    CROP = 4


    transform_list, transform_list_test = ListTransform()

    DG_TRAIN = DataGenRandomT(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file=None)

    test_patient = ["141549", "162438"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE

    DG_TEST  = DataGenRandomT(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(test_patient)

    model = UNetBatchNorm(TFRecord,    LEARNING_RATE=LEARNING_RATE,
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
