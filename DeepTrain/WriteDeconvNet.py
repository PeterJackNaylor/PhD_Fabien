import DeconvNet
import os

def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def WriteDeconvNet(kwargs):
    """
    Compulsory arguments:
      wd : work directory
      cn : classifier name


    Optionnal:
      loss : loss wished, specifically for the weight generations
      batch_size : size of the batches to give to the network
    """

    if 'loss' in kwargs.keys():
        loss = kwargs['loss']
        if "weight" in loss:
            loss = 'weight'
        print "loss ----   {}   -----".format(str(loss))

    else:
        loss = 'softmax'

    if 'batch_size' in kwargs.keys():
        batch_size = kwargs['batch_size']
    else:
        batch_size = 1

    path_ = os.path.join(kwargs['wd'], kwargs['cn'])
    path_modelgen = os.path.join(path_, "model")

    datagen_path = os.path.join(path_modelgen, "data_generator_train.pkl")
    datagen_test_path = os.path.join(path_modelgen, "data_generator_test.pkl")

    CheckOrCreate(os.path.join(path_))

    DeconvNet.make_net(path_,
                       datagen_path,
                       datagen_test_path,
                       batch_size=batch_size,
                       classifier_name=kwargs['cn'],
                       loss_layer=loss)
