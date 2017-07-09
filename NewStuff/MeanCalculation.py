import numpy as np
from DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import pdb
from os.path import join
from optparse import OptionParser


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("--path", dest="path",type="string",
                      help="path to annotated dataset")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")

    (options, args) = parser.parse_args()


    path = "/data/users/pnaylor/Bureau/ToAnnotate"
    path = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotate"


    path = options.path
    transf, transf_test = ListTransform()

    size = (224, 224)
    crop = 4
    DG = DataGenRandomT(path, crop=crop, size=size, transforms=transf_test,
                 split="train", num="")

    res = np.zeros(shape=(224, 224, 3, DG.length))
    for i in range(DG.length):
        key = DG.NextKeyRandList(0)
        res[:,:,:,i] = DG[key][0]

    mean = np.mean(res, axis=(0,1,3))
    np.save(join(options.out, "mean_file.npy"), mean)
