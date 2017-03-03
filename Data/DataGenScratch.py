import random
import itertools
import pdb
from DataGen import DataGen
from WrittingTiff.tifffile import imread
import os
from UsefulFunctions import ImageTransf as Transf
from UsefulFunctions.ImageTransf import Identity, flip_vertical, flip_horizontal
import matplotlib.pylab as plt
import numpy as np
import cPickle as pkl
from RandomUtils import CheckExistants, CheckFile





def MakeDataGen(options):
    dgtrain = options.dgtrain
    dgtest = options.dgtest

    rawdata = options.rawdata
    rawdata_lbl = rawdata.replace('volume', 'labels')
    leaveout = options.leaveout
    seed = options.seed
    Unet = options.net == "UNet"
    wgt_param = options.wgt_param
    transform_list = options.transform_list
    Weight = options.Weight
    WeightOnes = options.WeightOnes


    data_generator_train = DataGenIsbi2012(rawdata, rawdata_lbl,
                                  transform_list, split="train",
                                   leave_out=leaveout, seed=seed, Weight=Weight,
                                   WeightOnes=WeightOnes, wgt_param=wgt_param,
                                   Unet=Unet)
    pkl.dump(data_generator_train, open(dgtrain, "wb"))

    data_generator_test = DataGenIsbi2012(rawdata, rawdata_lbl,
                                  [Identity()], split="test",
                                  leave_out=leaveout, seed=seed, Weight=Weight,
                                  WeightOnes=WeightOnes, wgt_param=wgt_param,
                                  Unet=Unet)
    pkl.dump(data_generator_test, open(dgtest, "wb"))


def LoadSetImage(path):
    """This function is able to loads a multi page tif."""
    SetImage = imread(path)
    return SetImage


class DataGenScratch(DataGen):
    """
    DataGen object that can deal with the Camelyon dataset
    """
    def __init__(self, CamelyonTextFilesFolder, split, seed):



    def LoadFiles(self):

        txt_parameter = os.path.join(self.CamelyonTextFilesFolder, self.split, '.txt')
        table = pd.read_csv(txt_parameter, sep = " ", header = None)
        listoflist = table.values.tolist()
        if self.split = "test":
             return listoflist
        def f(*args):
            out = ""
            for el in args:
                out += str(el)
                out += "_"
            return out[:-1]
        only_str = map(f, listoflist)
        n_transfom = len(self.transform)
        all_combi = list(iter.tools.product(only_str, range(n_transfom)))
        def g(el, el2):
            list1 = el.split('_')
            list1 += el2
            return list1
        listoflist2 = map(g, all_combi)
        return listoflist2

            
