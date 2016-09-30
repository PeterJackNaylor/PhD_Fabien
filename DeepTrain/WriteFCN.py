import os


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def WriteFCN(kwargs):
    """
    Compulsory arguments:
      wd : work directory
      cn : classifier name
      archi: list of architectures [8, 16, 32]

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
    print 'architectures :' + str(kwargs["archi"])

    path_ = os.path.join(kwargs['wd'], kwargs['cn'])
    path_modelgen = os.path.join(path_, "model")

    datagen_path = os.path.join(path_modelgen, "data_generator_train.pkl")
    datagen_test_path = os.path.join(path_modelgen, "data_generator_test.pkl")

    CheckOrCreate(os.path.join(path_))
    for arc in kwargs["archi"]:
        if arc == 8:
            import FCN8
            if len(kwargs["archi"]) == 1:
                temp_path_ = path_
            else:
                temp_path_ = os.path.join(path_, "FCN8")
            FCN8.make_net(temp_path_,
                          datagen_path,
                          datagen_test_path,
                          batch_size=batch_size,
                          classifier_name=kwargs['cn'],
                          loss_layer=loss)
        if arc == 16:
            import FCN16
            if len(kwargs["archi"]) == 1:
                temp_path_ = path_
            else:
                temp_path_ = os.path.join(path_, "FCN16")
            FCN16.make_net(temp_path_,
                           datagen_path,
                           datagen_test_path,
                           batch_size=batch_size,
                           classifier_name=kwargs['cn'],
                           loss_layer=loss)
        if arc == 32:
            import FCN32
            if len(kwargs["archi"]) == 1:
                temp_path_ = path_
            else:
                temp_path_ = os.path.join(path_, "FCN32")
            FCN32.make_net(temp_path_,
                           datagen_path,
                           datagen_test_path,
                           batch_size=batch_size,
                           classifier_name=kwargs['cn'],
                           loss_layer=loss)
