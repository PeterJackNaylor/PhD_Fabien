import numpy as np
from optparse import OptionParser
from os.path import join
from UsefulFunctions.RandomUtils import CheckOrCreate

list_f = [np.min, np.mean, np.std, np.max]

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table_name",type="string",
                      help="table name")
    parser.add_option("--feat", dest="feat", type="int",
                      help="int corresponding to the feature to visualize")
    parser.add_option("--output", dest="out", type="string",
                      help="output localisation")
    (options, args) = parser.parse_args()

    name = options.table_name
    feat = options.feat

    table = np.load(name)
    vect = table[:, feat]


    metrics = np.zeros(len(list_f))
    for i, func in enumerate(list_f):
        metrics[i] = func(vect)

    save_name = name.replace(".npy", "_{}_color_0.npy").format(feat)
    CheckOrCreate(options.out)
    save_name = join(options.out, save_name)
    np.save(save_name, metrics)