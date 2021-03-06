# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pkl
import os
from optparse import OptionParser
from skimage.io import imsave
#  predicting from ensemble/ dictionnary of nets


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-o", "--output", dest="output",
                      help="Output folder")

    (options, args) = parser.parse_args()

    # checking the input data

    CheckOrCreate(options.output)
    CheckOrCreate(os.path.join(options.output, "NotEnlarge"))
    CheckOrCreate(os.path.join(options.output, "Enlarge"))

    datagen_not = "/data/users/pnaylor/Documents/Python/LoopingDeconvNetFromPretrainedWeight2/DeconvNet_1_0.9_0.0005/model/data_generator_train.pkl"
    datagen_enlarge = datagen_not
    datagen_not = pkl.load(open(datagen_not, "r"))
    datagen_enlarge = pkl.load(open(datagen_enlarge, "r"))
    datagen_not.Weight = False
    pat = 4
    sli = 3
    crop = 3
    print "Input paramters to OutputNet:"
    print " \n "
    print "Output folder     : | " + options.output
    print ' \n '
    print "Beginning analyse:"
    for i in range(len(datagen_enlarge.transforms)):
        list_img = datagen_enlarge[pat, sli, i, crop]
        patient = os.path.join(options.output, "Enlarge/Enlarge{}.png")
        patient_gt = os.path.join(
            options.output, "Enlarge/Enlarge{}_gt.png")
        imsave(patient.format(i), list_img[0])
	print np.unique(list_img[1]), list_img[1].shape
        try:
            gt = list_img[1][:, :, 0]
        except:
            gt = list_img[1]
        imsave(patient_gt.format(i), gt)
