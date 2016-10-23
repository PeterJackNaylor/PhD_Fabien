# -*- coding: utf-8 -*-

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

    datagen_not = "/data/users/pnaylor/Documents/Python/"
    datagen_enlarge = "/data/users/pnaylor/Documents/Python/LoopingDeconvNetFromPretrainedWeight/DeconvNet_0.1_0.9_0.0005/model/data_generator_train.pkl"

    datagen_not = pkl.load(open(datagen_not, "r"))
    datagen_enlarge = pkl.load(open(datagen_enlarge, "r"))
    datagen_not.Weight = False
    pat = 0
    sli = 1
    crop = 0
    print "Input paramters to OutputNet:"
    print " \n "
    print "Output folder     : | " + options.output
    print ' \n '
    print "Beginning analyse:"

    key = datagen_not.RandomKey(True)
    for i in range(len(datagen_not.transforms)):
        key = datagen_not.NextKeyRandList(key)
        list_img = datagen_not[key]
        patient = os.path.join(options.output, "{}.png")
        patient_gt = os.path.join(
            options.output, "{}_gt.png")
        imsave(patient.format(i), list_img[0])
        try:
            gt = list_img[1][:, :, 0]
        except:
            gt = list_img[1]
        imsave(patient_gt.format(i), gt)
"""
    for i in range(len(datagen_enlarge.transforms)):
        list_img = datagen_enlarge[pat, sli, i, crop]
        patient = os.path.join(options.output, "Enlarge/Enlarge{}.png")
        patient_gt = os.path.join(
            options.output, "Enlarge/Enlarge{}_gt.png")
        imsave(patient.format(i), list_img[0])
	try:
	    gt = list_img[1][:,:,0]
	except:
	    gt = list_img[1]
        imsave(patient_gt.format(i), gt)
"""
