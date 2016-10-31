import ImageTransf as Transf
import os
import cPickle as pkl
from DataGen import DataGen
import numpy as np


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def WriteDataGen(kwargs):
    """
    Compulsory arguments:
        wd : work directory
        cn : classifier name
        rawdata: where the raw data is
        val_num: number of patients to leave out for test


    Optionnal:
        enlarge : if to enlarge image when deforming
        loss : loss wished, specifically for the weight generations
        crop : if to crop
        crop_size : if random cropping
        img_format : can be RGB, HE, HEDab
        seed: seed to pick from


    """

    wd = kwargs['wd']
    cn = kwargs['cn']
    rawdata = kwargs['rawdata']
    leaveout = int(kwargs['val_num'])
    Unet = kwargs["UNet"]

    path_modelgen = os.path.join(wd, cn, "model")

    CheckOrCreate(path_modelgen)

    print 'wd        ----   {}   ------'.format(str(wd))
    print 'cn        ----   {}   ------'.format(str(cn))
    print 'leaveout  ----   {}   ------'.format(str(leaveout))
    print 'rawdata   ----   {}   ------'.format(str(rawdata))

    if 'enlarge' in kwargs.keys():
        enlarge = kwargs['enlarge']
        print "Enlarge ----   {}   -----".format(str(enlarge))
    else:
        enlarge = False

    Weight = False
    WeightOnes = False
    if 'loss' in kwargs.keys():
        loss = kwargs['loss']
        print "loss    ----   {}   -----".format(str(loss))
        if loss == "weight" or loss == "weightcpp":
            Weight = True
            WeightOnes = False
        elif loss == "weight1" or loss == "weightcpp1":
            Weight = True
            WeightOnes = True

    else:
        loss = 'softmax'

    if 'crop' in kwargs.keys():
        crop = kwargs['crop']
        print "crop    ----   {}   -----".format(str(crop))
    else:
        crop = None

    if 'crop_size' in kwargs.keys():
        crop_size = kwargs['crop_size']
        print "crop_size ----   {}   -----".format(str(crop_size))
    else:
        crop_size = None

    if 'img_format' in kwargs.keys():
        img_format = kwargs['img_format']
        print "img_format----   {}   -----".format(str(img_format))

    else:
        img_format = "RGB"

    if 'seed' in kwargs.keys():
        seed = kwargs['seed']
        print "seed      ----   {}   -----".format(str(seed))

    else:
        seed = 42

    transform_list = [Transf.Identity(),
                      Transf.Flip(0),
                      Transf.Flip(1)]

    for rot in np.arange(1, 360, 4):
        transform_list.append(Transf.Rotation(rot, enlarge=enlarge))

    for sig in [1, 2, 3, 4]:
        transform_list.append(Transf.OutOfFocus(sig))

    for i in range(50):
        transform_list.append(Transf.ElasticDeformation(0, 12, num_points=4))

    data_generator_train = DataGen(rawdata, crop=crop, size=crop_size,
                                   transforms=transform_list, split="train",
                                   leave_out=leaveout, seed=seed,
                                   img_format=img_format, Weight=Weight,
                                   WeightOnes=WeightOnes, Unet=Unet)

    pkl.dump(data_generator_train, open(os.path.join(
        path_modelgen, "data_generator_train.pkl"), "wb"))
    data_generator_test = DataGen(rawdata, crop=crop, size=crop_size,
                                  transforms=[Transf.Identity()], split="test",
                                  leave_out=leaveout, seed=seed,
                                  img_format=img_format, Weight=Weight,
                                  WeightOnes=WeightOnes, Unet=Unet)
    pkl.dump(data_generator_test, open(os.path.join(
        path_modelgen, "data_generator_test.pkl"), "wb"))
