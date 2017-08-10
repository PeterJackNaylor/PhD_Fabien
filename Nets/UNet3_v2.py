# -*- coding: utf-8 -*-
### for objects
from UNetMultiClass_v2 import UNetMultiClass
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
### for main
from optparse import OptionParser
from Data.DataGenClass import DataGen3
from UsefulFunctions.ImageTransf import ListTransform
import pdb

if __name__== "__main__":

    parser = OptionParser()

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

    TFRecord = options.TFRecord
    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    DROPOUT = options.dropout
    MEAN_FILE = options.mean_file 
    N_THREADS = options.THREADS


    LEARNING_RATE = options.lr
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
    CROP = 4
    #pdb.set_trace()
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str


    transform_list, transform_list_test = ListTransform()

    DG_TRAIN = DataGen3(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file=None)

    test_patient = ["141549", "162438"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE
    DG_TEST  = DataGen3(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(test_patient)

    model = UNetMultiClass(TFRecord,   LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_LABELS=3,
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
    lb = ["Background", "Nuclei", "NucleiBorder"]
    model.train(DG_TEST, lb_name=lb)
    
