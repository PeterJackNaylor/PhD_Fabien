import numpy as np
from optparse import OptionParser
from GetMax import list_f
import glob
from os.path import join, basename

def BlueRedGrad(val, min_val, max_val):
    alpha = float(val - min_val) / float(max_val - min_val)
    RGB = np.array([int(alpha * 255), 0, int((1 - alpha) * 255)])
    return RGB

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table",type="string",
                      help="table")

    (options, args) = parser.parse_args()
    METRICS = glob.glob('METRIC_GENERAL_*.npy')
    for met in METRICS:
        table = np.load(options.table)
        metrics = np.load(met)

        feat = int(met.split('_')[-1].split(".")[0])
        vect = table[:, feat]
        def f(val):
            return BlueRedGrad(val, metrics[0], metrics[3])
        res = map(f, vect)
        name_res = basename(options.table).replace(".npy", "_feat_{}_color.npy").format(feat)
        np.save(name_res, res)