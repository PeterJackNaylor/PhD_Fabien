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
    SIZE = (options.size, options.size)
    N_ITER_MAX = 0 ## defined later
    N_TRAIN_SAVE = 1000
    MEAN_FILE = options.mean_file 
    save_dir = options.log
    checkpoint = options.restore

    model = FCN8(checkpoint, save_dir, TFRecord, options.size,
                 2, 1000)

    if SPLIT == "train":
        model.train8(N_ITER_MAX, LEARNING_RATE)
    elif SPLIT == "test":
        model.test8()
    elif SPLIT == "validation":
        model.validate()
