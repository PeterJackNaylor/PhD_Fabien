import numpy as np
from optparse import OptionParser
from GetStatistics4Color import list_f, CheckOrCreate
import glob
from os.path import join, basename
from skimage.measure import label
import pandas as pd
from WrittingTiff.Extractors import list_f_names
from tifffile import imsave, imread
import pdb

def DivergingPurpleGreen(rank, max_rank):
    """ Have to write where they go and what they do"""
    purple_4 = np.array([64,0,75])
    purple_3 = np.array([118,42,131])
    purple_2 = np.array([153,112,171])
    purple_1 = np.array([194,165,207])
    purple_0 = np.array([231,212,232])
    white = np.array([247,247,247])
    green_0 = np.array([217,240,211])
    green_1 = np.array([166,219,160])
    green_2 = np.array([90,174,97])
    green_3 = np.array([27,120,55])
    green_4 = np.array([0,68,27])

def SequentialPurple(rank, max_rank):
    white_3 = np.array([247,252,253])
    white_2 = np.array([224,236,244])
    white_1 = np.array([191,211,230])
    white_0 = np.array([158,188,218])
    middle = np.array([140,150,198])
    purple_0 = np.array([140,107,177])
    purple_1 = np.array([136,65,157])
    purple_2 = np.array([129,15,124])
    purple_3 = np.array([77,0,75])




def BlueRedGrad(val, min_val, max_val):
    alpha = float(val - min_val) / float(max_val - min_val)
    RGB = np.array([int(alpha * 255), 0, int((1 - alpha) * 255)])
    return RGB

def ClosestLabel(binary, x, y):
    x = int(x)
    y = int(y)
    max_x , max_y = binary.shape[0:2] 
    if binary[x, y] != 0:
        return binary[x, y]
    else:
        found = False
        dist = 0
        while not found:
            dist += 1
	    MIN_X = min(max_x - 1, max(0, x-dist))
	    MAX_X = min(max_x - 1, max(0, x+dist))
	    
	    MAX_Y = min(max_y - 1, max(0, y+dist))
	    MIN_Y = min(max_y - 1, max(0, y-dist))
	
            possible_labels = np.unique(binary[MIN_X:MAX_X, MIN_Y:MAX_Y])
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
    CheckOrCreate(options.out)
    x, y = bin.shape
    table = pd.DataFrame(np.load(options.table), columns=list_f_names)
    table = table[(table.T != 0).any()]
    if table.shape[0] == 0:
        for met in METRICS:
            feat = int(met.split('_')[-1].split('.')[0])
            out = join(options.out, "feat_" + list_f_names[feat])
            CheckOrCreate(out)
            color_copy = np.zeros(shape=(x,y,3), dtype='uint8')
            imsave(join(out, basename(options.table).replace("npy", "tiff")), color_copy, resolution=[1.0,1.0]) 
    else:
        for met in METRICS:
            color_copy = np.zeros(shape=(x,y,3), dtype='uint8')
            metrics = np.load(met)
            feat = int(met.split('_')[-1].split('.')[0])
            def f(val, x, y):
                label = ClosestLabel(bin, x, y)
                color_copy[bin == label] = BlueRedGrad(val, metrics[0], metrics[3])

            table.apply(lambda row: f(row[list_f_names[feat]], row["Centroid_x"], row["Centroid_y"]), axis=1)
            out = join(options.out, "feat_" + list_f_names[feat])
            CheckOrCreate(out)
            imsave(join(out, basename(options.table).replace("npy", "tiff")), color_copy, resolution=[1.0,1.0])
