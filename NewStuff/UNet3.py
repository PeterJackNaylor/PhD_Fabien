# -*- coding: utf-8 -*-
### for objects
from UNetMultiClass import UNetMultiClass
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
### for main
from optparse import OptionParser
from DataGenClass import DataGen3
from UsefulFunctions.ImageTransf import ListTransform
import pdb

if __name__== "__main__":

    parser = OptionParser()
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
    (options, args) = parser.parse_args()

    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    LRSTEP = "10epoch"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch
    PATH = options.path
    HEIGHT = 212
    WIDTH = 212
    N_TRAIN_SAVE = 2
    CROP = 4
    #pdb.set_trace()
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str


    transform_list, transform_list_test = ListTransform()

    DG_TRAIN = DataGen3(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file="mean_file.npy")

    test_patient = ["141549", "162438"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE
    DG_TEST  = DataGen3(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file="mean_file.npy")
    DG_TEST.SetPatient(test_patient)

    model = UNetMultiClass(LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=HEIGHT,
                                       NUM_LABELS=3,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES)
    lb = ["Background", "Nuclei", "NucleiBorder"]
    model.train(DG_TRAIN, DG_TEST, lb_name=lb)
    
