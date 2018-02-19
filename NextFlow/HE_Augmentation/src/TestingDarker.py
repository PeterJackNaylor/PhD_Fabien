from optparse import OptionParser
import tensorflow as tf 
from scipy.misc import imread, imsave
from os.path import join, basename
import numpy as np
from Nets.UNetBatchNorm_v2 import UNetBatchNorm
from Deprocessing.Morphology import PostProcess
from UsefulFunctions.RandomUtils import CheckOrCreate
from glob import glob
import pdb
from Data.DataGenClass import DataGenMulti
from UsefulFunctions.ImageTransf import ListTransform
import math
from Testing import bettermodel, Options, parse_meta_file, find_latest



if __name__ == '__main__':
    options = Options()

    ### PUT DATAGEN
    MEAN_FILE = options.mean_file 
    transform_list, transform_list_test = ListTransform()
    output = options.output

    TEST_PATIENT = ["bladder", "colorectal", "stomach"]
    x, y, c = (996, 996, 3)

    DG_TEST  = DataGenMulti(options.path, split="test", crop = 1, size=(x, y), num=TEST_PATIENT,
                       transforms=transform_list_test, seed_=42, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(TEST_PATIENT)


    stepSize = x
    windowSize = (x + 184, y + 184)
    META = find_latest(options.folder)
    n_features = int(basename(options.folder).split('_')[0])

    model = bettermodel("f",
                          BATCH_SIZE=1, 
                          IMAGE_SIZE = (x, y),
                          NUM_CHANNELS=c, 
                          NUM_LABELS=2,
                          N_FEATURES=n_features,
                          LOG=options.folder)

    l, acc, F1, recall, precision, meanacc, AJI = model.Validation(DG_TEST)

    file_name = output + ".txt"
    f = open(file_name, 'w')
    f.write('AJI: # {} #\n'.format(AJI))
    f.write('Mean acc: # {} #\n'.format(meanacc))
    f.write('Precision: # {} #\n'.format(precision))
    f.write('Recall: # {} #\n'.format(recall))
    f.write('F1: # {} #\n'.format(F1))
    f.write('ACC: # {} #\n'.format(acc)) 
    f.close()
