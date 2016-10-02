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
    """
    if 'loss' in kwargs.keys():
        loss = kwargs['loss']
        if "weightcpp" in loss:
            loss = 'weightcpp'
        elif "weight" in loss:
            loss = 'weight'
        print "loss ----   {}   -----".format(str(loss))

    else:
        loss = 'softmax'

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
                temp_path_ = path_
            FCN8.make_net(temp_path_,
                          datagen_path,
                          datagen_test_path,
                          classifier_name=kwargs['cn'],
                          classifier_name1="score_fr_2",
                          classifier_name2="upscore2",
                          classifier_name3="score_pool4",
                          loss_layer=loss)
        if arc == 16:
            import FCN16
            if len(kwargs["archi"]) == 1:
                temp_path_ = path_
            else:
                temp_path_ = path_
            FCN16.make_net(temp_path_,
                           datagen_path,
                           datagen_test_path,
                           classifier_name=kwargs['cn'],
                           classifier_name1="score_fr_2",
                           loss_layer=loss)
        if arc == 32:
            import FCN32
            if len(kwargs["archi"]) == 1:
                temp_path_ = path_
            else:
                temp_path_ = path_
            FCN32.make_net(temp_path_,
                           datagen_path,
                           datagen_test_path,
                           classifier_name=kwargs['cn'],
                           classifier_name1="score_fr_2",
                           classifier_name2="upscore_2",
                           loss_layer=loss)
