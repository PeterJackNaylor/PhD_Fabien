import numpy as np
from optparse import OptionParser
from GetStatistics4Color import CheckOrCreate, list_f
import glob
from os.path import join
import pdb
if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--path", dest="path",type="string",
                      help="path")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")
    parser.add_option("--key", dest="key",type="string",
                      help="patient id")
    (options, args) = parser.parse_args()

    path_npy = join(options.path, "*_color_0.npy")

    
    N_feat = []
    for f in glob.glob(path_npy):
        feat_num = f.split('_')[-3]
        if feat_num not in N_feat:
            N_feat.append(feat_num)
#    pdb.set_trace()
    CheckOrCreate(options.out)
    dic_feat = {n_feat: join(options.path, "*_{}_color_0.npy".format(n_feat)) for n_feat in N_feat}
    for n_feat in N_feat:
        feat_table = glob.glob(dic_feat[n_feat])

        table_new = np.zeros(shape=(len(feat_table), len(list_f)))
        for i, name in enumerate(feat_table):
            table_new[i, :] = np.load(name)

        metrics = np.zeros(len(list_f))
        for i, func in enumerate(list_f):
            #pdb.set_trace()
            metrics[i] = func(table_new[:, i])


        save_name = "GeneralStatistics4color_{}_{}.npy".format(options.key, n_feat)
        save_name = join(options.out, save_name)
        np.save(save_name, metrics)