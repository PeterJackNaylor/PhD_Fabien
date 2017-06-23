import numpy as np
from optparse import OptionParser
from GetMax import list_f
import glob
from os.path import join
import pdb
if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--path", dest="path",type="string",
                      help="path")

    (options, args) = parser.parse_args()

    path_npy = join(options.path, "*.npy")

    
    N_feat = []
    for f in glob.glob(path_npy):
        feat_num = f.split('_')[-1].split('.')[0]
        if feat_num not in N_feat:
            N_feat.append(feat_num)
#    pdb.set_trace()
    dic_feat = {n_feat: join(options.path, "*_{}.npy".format(n_feat)) for n_feat in N_feat}
    for n_feat in N_feat:
        feat_table = glob.glob(dic_feat[n_feat])

        table_new = np.zeros(shape=(len(feat_table), len(list_f)))
        for i, name in enumerate(feat_table):
            table_new[i, :] = np.load(name)

        metrics = np.zeros(len(list_f))
        for i, func in enumerate(list_f):
            #pdb.set_trace()
            metrics[i] = func(table_new[:, i])

        save_name = "METRIC_GENERAL_{}.npy".format(n_feat)
        np.save(save_name, metrics)