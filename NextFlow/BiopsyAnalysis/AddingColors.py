import numpy as np
from optparse import OptionParser
from GetStatistics4Color import list_f, CheckOrCreate
import glob
from os.path import join, basename
from scipy.misc import imread
from skimage.measure import label
import pandas as pd
from WrittingTiff.Extractors import list_f_names, CheckOrCreate
from tifffile import imsave


def BlueRedGrad(val, min_val, max_val):
    alpha = float(val - min_val) / float(max_val - min_val)
    RGB = np.array([int(alpha * 255), 0, int((1 - alpha) * 255)])
    return RGB

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table",type="string",
                      help="table")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")
    parser.add_option("--key", dest="key",type="string",
                      help="patient id")

    (options, args) = parser.parse_args()
    METRICS = glob.glob('Job_{}/GeneralStats4Color/GeneralStatistics4color_*.npy'.format(options.key))
    bin = label(imread("Job_{}/bin/{}.npy".format(options.key, options.table.replace('.npy', '.tiff'))))
    CheckOrCreate(options.output)
    x, y = bin.shape

    for met in METRICS:
        color_copy = np.zeros(shape=(x,y,3), type='float64')

        metrics = np.load(met)
        feat = int(met.split('_')[-1].split('.')[0])
        table = pd.DataFrame(options.table, columns=list_f_names)
        def f(val, x, y):
            label = bin[x, y]
            color_copy[bin == label] = BlueRedGrad(val, metrics[0], metrics[3])
        table.apply(lambda row: f(row[list_f_names[feat], row["Centroid_x"], row["Centroid_y"]]))
        out = join(options.output, list_f_names[feat])
        CheckOrCreate(out)
        imsave(join(out, basename(options.table) + ".tiff"), color_copy)
