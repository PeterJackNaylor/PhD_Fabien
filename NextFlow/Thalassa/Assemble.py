import numpy as np
from optparse import OptionParser
from GetMax import list_f
import glob
from os.path import join

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--path", dest="path",type="string",
                      help="path")

    (options, args) = parser.parse_args()

    path_npy = join(options.path, "*.npy")

    table_new = np.zeros(shape=(len(path_npy), len(list_f)))
    for i, name in enumerate(glob.glob(path_npy)):
    	table_new[i, :] = np.load(name)

    metrics = np.zeros(len(list_f))
    for i, func in enumerate(list_f):
        metrics[i] = func(table_new[:, i])

    save_name = "METRIC_GENERAL.npy"
    np.save(save_name, metrics)