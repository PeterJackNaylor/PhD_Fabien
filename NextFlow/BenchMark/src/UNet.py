# -*- coding: utf-8 -*-
from utils import GetOptions, ComputeMetrics
from glob import glob
from Nets.UNetBatchNorm_v2 import UNetBatchNorm
import tensorflow as tf
import numpy as np
import os
from UsefulFunctions.RandomUtils import CheckOrCreate
from Data.DataGenClass import DataGenMulti
from UsefulFunctions.ImageTransf import ListTransform
import math
import pdb



class Model(UNetBatchNorm):
    def test(self, p1, p2, steps):
        loss, roc = 0., 0.
        acc, F1, recall = 0., 0., 0.
        precision, jac, AJI = 0., 0., 0.
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        self.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(steps):  
            feed_dict = {self.is_training: False} 
            l,  prob, batch_labels = self.sess.run([self.loss, self.train_prediction,
                                                               self.train_labels_node], feed_dict=feed_dict)
            loss += l
            out = ComputeMetrics(prob[0,:,:,1], batch_labels[0,:,:,0], p1, p2)
            acc += out[0]
            roc += out[1]
            jac += out[2]
            recall += out[3]
            precision += out[4]
            F1 += out[5]
            AJI += out[6]
        coord.request_stop()
        coord.join(threads)
        loss, acc, F1 = np.array([loss, acc, F1]) / steps
        recall, precision, roc = np.array([recall, precision, roc]) / steps
        jac, AJI = np.array([jac, AJI]) / steps
        return loss, acc, F1, recall, precision, roc, jac, AJI

    def validation(self, DG_TEST, p1, p2, save_path):
        n_test = DG_TEST.length
        n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 
        loss, roc = [], []
        acc, F1, recall = [], [], []
        precision, jac, AJI = [], [], []
        res = []

        for i in range(n_batch):
            Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l, pred = self.sess.run([self.loss, self.train_prediction],
                                    feed_dict=feed_dict)
            rgb = (Xval[0,92:-92,92:-92] + np.load(self.MEAN_FILE)).astype(np.uint8)
            out = ComputeMetrics(pred[0,:,:,1], Yval[0,:,:,0], p1, p2, rgb=rgb, save_path=save_path, ind=i)
            out = [l] + list(out)
            res.append(out)
        return res


if __name__== "__main__":

    transform_list, transform_list_test = ListTransform()
    options = GetOptions()

    SPLIT = options.split

    ## Model parameters
    TFRecord = options.TFRecord
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    SIZE = (options.size_train, options.size_train)
    if options.size_test is not None:
        SIZE = (options.size_test, options.size_test)
    N_ITER_MAX = 0 ## defined later
    LRSTEP = "10epoch"
    N_TRAIN_SAVE = 100
    LOG = options.log
    WEIGHT_DECAY = options.weight_decay
    N_FEATURES = options.n_features
    N_EPOCH = options.epoch
    N_THREADS = options.THREADS
    MEAN_FILE = options.mean_file 
    DROPOUT = options.dropout

    ## Datagen parameters
    PATH = options.path
    TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach", "test"]
    DG_TRAIN = DataGenMulti(PATH, split='train', crop = 16, size=SIZE,
                       transforms=transform_list, UNet=True, mean_file=None, num=TEST_PATIENT)
    TEST_PATIENT = ["test"]
    DG_TEST  = DataGenMulti(PATH, split="test", crop = 1, size=(500, 500), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE, num=TEST_PATIENT)
    if SPLIT == "train":
        N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE
    elif SPLIT == "test":
        N_ITER_MAX = N_EPOCH * DG_TEST.length // BATCH_SIZE
    elif SPLIT == "validation":
        LOG = glob(os.path.join(LOG, '*'))[0] 
    model = Model(TFRecord,            LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=LOG,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=DROPOUT)
    if SPLIT == "train":
        model.train(DG_TEST)
    elif SPLIT == "test":
        p1 = options.p1
        file_name = options.output
        f = open(file_name, 'w')
        outs = model.test(options.p1, 0.5, N_ITER_MAX)
        outs = [LOG] + list(outs) + [p1, 0.5]
        NAMES = ["ID", "Loss", "Acc", "F1", "Recall", "Precision", "ROC", "Jaccard", "AJI", "p1", "p2"]
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*NAMES))

        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*outs))

    elif SPLIT == "validation":

        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach"]

        file_name = options.output
        f = open(file_name, 'w')
        NAMES = ["NUMBER", "ORGAN", "Loss", "Acc", "F1", "Recall", "Precision", "ROC", "Jaccard", "AJI", "p1", "p2"]
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*NAMES))


        for organ in TEST_PATIENT:
            DG_TEST  = DataGenMulti(PATH, split="test", crop = 1, size=(996, 996),num=[organ],
                           transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
            save_organ = os.path.join(options.save_path, organ)
            CheckOrCreate(save_organ)
            outs = model.validation(DG_TEST, options.p1, 0.5, save_organ)
            for i in range(len(outs)):
                small_o = outs[i]
                small_o = [i, organ] + small_o + [options.p1, 0.5]
                f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*small_o))
        f.close()

