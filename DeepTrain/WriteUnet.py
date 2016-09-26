import Unet


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def WriteUnet(**kwargs):
    """
    Compulsory arguments:
        wd : work directory
        cn : classifier name

    Optionnal:
        loss : loss wished, specifically for the weight generations
    """

    if 'loss' in kwargs.keys():
        loss = kwargs['loss']
        if "weight" in loss:
            loss = 'weight'
        print "loss ----   {}   -----".format(str(loss))

    else:
        loss = 'softmax'

    path_ = os.path.join(kwargs['wd'], kwargs['cn'])
    path_modelgen = os.path.join(path_, "model")

    datagen_path = os.path.join(path_modelgen, "data_generator_train.pkl")

    CheckOrCreate(os.path.join(path_))

    Unet.make_net(path_,
                  datagen_path,
                  os.path.join(path_modelgen, "data_generator_test.pkl"),
                  classifier_name=kwargs['cn'],
                  loss_layer=loss)
