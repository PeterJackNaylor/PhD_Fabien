import Unet
import os
import BaochuanNet
import pdb


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def WriteBaochuanNet(kwargs):
    """
    Compulsory arguments:
        wd : work directory
        cn : classifier name
        batch_size: batch size for trai 
    """
    pdb.set_trace()
    batch_size = kwargs['batch_size']

    path_ = os.path.join(kwargs['wd'], kwargs['cn'])
    path_modelgen = os.path.join(path_, "model")

    datagen_path = os.path.join(path_modelgen, "data_generator_train.pkl")
    datagen_test = os.path.join(path_modelgen, "data_generator_test.pkl")
    CheckOrCreate(os.path.join(path_))
    BaochuanNet.make_net(path_,
                         datagen_path,
                         datagen_test,
                         classifier_name=kwargs['cn'],
                         batch_size=batch_size)
