import os
import cPickle as pkl


def find_between_r(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def ChangeDataGenPath(path, folder):
    """
    Sets the datagenerator in folder so that they can find
    the raw images in path.
    Input:
        path: raw image folder
        folder: path to working folder
    """
    datagen_test = os.path.join(folder, 'model', 'data_generator_test.pkl')
    datagen_train = os.path.join(folder, 'model', 'data_generator_train.pkl')

    dg_train = pkl.load(open(datagen_train, "rb"))
    dg_test = pkl.load(open(datagen_test, "rb"))

    dg_train.SetPath(path)
    dg_test.SetPath(path)

    pkl.dump(dg_train, open(datagen_train, "w"))
    pkl.dump(dg_test, open(datagen_test, "w"))


def ChangePrototxt(protofile, folder):
    """
    Modifyies the protofile so that he know where to find the datagen
    in the right folder.
    Input:
        protofile: protofile name
        folder: path to working folder
    """
    with open(protofile, 'r') as file:
        filedata = file.read()

    to_be_replaced = find_between_r(filedata, "datagen\\': \\'", '.pkl')
    split = to_be_replaced.split('_')[-1]
    to_replace = os.path.join(
        folder, "model", "data_generator_{}".format(split))
    filedata = filedata.replace(to_be_replaced, to_replace)

    with open(protofile, 'w') as file:
        file.write(filedata)


def ChangeEnv(path, folder):
    """
    Changes the setting for a whole caffe model, which means 
    relacing the data generators so that they find path correctly
    but also know where all the other related files are by specifying
    folder.
    Input: 
        path: raw image folder
        folder: path to working folder
    """
    ChangeDataGenPath(path, folder)
    if 'FCN' in folder:
        for num in [32, 16, 8]:
            fcn_num = "FCN{}".format(num)
            protofile = os.path.join(folder, fcn_num, "train.prototxt")
            ChangePrototxt(protofile, folder)
            protofile = os.path.join(folder, fcn_num, "test.prototxt")
            ChangePrototxt(protofile, folder)
    else:
        protofile = os.path.join(folder, "train.prototxt")
        ChangePrototxt(protofile, folder)
        protofile = os.path.join(folder, "test.prototxt")
        ChangePrototxt(protofile, folder)
