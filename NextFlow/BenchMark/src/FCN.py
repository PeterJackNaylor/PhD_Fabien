from FCN_Object import FCN8
from utils import GetOptions, ComputeMetrics
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from datetime import datetime
from optparse import OptionParser
from UsefulFunctions.RandomUtils import CheckOrCreate
from UsefulFunctions.ImageTransf import ListTransform
from Data.CreateTFRecords import read_and_decode
import pdb
import os

if __name__== "__main__":

    options = GetOptions()

    SPLIT = options.split

    ## Model parameters
    TFRecord = options.TFRecord
    LEARNING_RATE = options.lr
    SIZE = (options.size_train, options.size_train)
    if options.size_test is not None:
        SIZE = (options.size_test, options.size_test)
    N_ITER_MAX = options.iters
    N_TRAIN_SAVE = 1000
    MEAN_FILE = options.mean_file 
    save_dir = options.log
    checkpoint = options.restore

    model = FCN8(checkpoint, save_dir, TFRecord, SIZE[0],
                 2, 1000)

    if SPLIT == "train":
        model.train8(N_ITER_MAX, LEARNING_RATE)
    elif SPLIT == "test":
        p1 = options.p1
        LOG = options.log

        file_name = options.output
        f = open(file_name, 'w')

        checkpoint = os.path.join(checkpoint, checkpoint) 
        outs = model.test8(N_ITER_MAX, checkpoint, p1)
        outs = [LOG] + list(outs) + [p1, 0.5]
        NAMES = ["ID", "Loss", "Acc", "F1", "Recall", "Precision", "ROC", "Jaccard", "AJI", "p1", "p2"]
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*NAMES))

        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*outs))
    elif SPLIT == "validation":
        model.validate()
