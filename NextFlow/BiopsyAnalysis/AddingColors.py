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

def ClostestLabel(binary, x, y):
    x = int(x)
    y = int(y)
    if binary[x, y] != 0:
        return binary[x, y]
    else:
        found = False
        dist = 0
        while not found:
            dist += 1
            possible_labels = np.unique(binary[(x-dist):(x+dist), (y-dist):(y+dist)])
            if len(possible_labels) != 1:
                found = True
        return possible_labels[1]

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
    bin = label(imread("Job_{}/bin/{}".format(options.key, options.table.replace('.npy', '.tiff'))))
    CheckOrCreate(options.output)
    x, y = bin.shape

    for met in METRICS:
        color_copy = np.zeros(shape=(x,y,3), type='float64')

        metrics = np.load(met)
        feat = int(met.split('_')[-1].split('.')[0])
        table = pd.DataFrame(options.table, columns=list_f_names)

        def f(val, x, y):
            label = ClosestLabel(bin, x, y)
            color_copy[bin == label] = BlueRedGrad(val, metrics[0], metrics[3])

        table.apply(lambda row: f(row[list_f_names[feat]], row["Centroid_x"], row["Centroid_y"]), axis=1)
        out = join(options.output, "feat_" + list_f_names[feat])
        CheckOrCreate(out)
        imsave(join(out, basename(options.table) + ".tiff"), color_copy)
