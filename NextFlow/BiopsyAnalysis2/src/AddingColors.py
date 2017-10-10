import numpy as np
from optparse import OptionParser
from WrittingTiff.Extractors import list_f_names
from UsefulFunctions.RandomUtils import CheckOrCreate
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
    a = (float(rank) / float(max_rank)) * 100
    if a < 9:
        return green_4
    elif a < 18:
        return green_3
    elif a < 27:
        return green_2
    elif a < 36:
        return green_1
    elif a < 45:
        return green_0
    elif a < 54:
        return white
    elif a < 63:
        return purple_0
    elif a < 72:
        return purple_1
    elif a < 81:
        return purple_2
    elif a < 90:
        return purple_3
    elif a < 101:
        return purple_4
    else:
        print "Problem in color scheme"

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

    a = (float(rank) / float(max_rank)) * 100
    if a < 11:
        return white_3
    elif a < 22:
        return white_2
    elif a < 33:
        return white_1
    elif a < 44:
        return white_0
    elif a < 55:
        return middle
    elif a < 66:
        return purple_0
    elif a < 77:
        return purple_1
    elif a < 88:
        return purple_2
    elif a < 101:
        return purple_3
    else:
        print "Problem in color scheme"




def BlueRedGrad(val, min_val, max_val):
    alpha = float(val - min_val) / float(max_val - min_val)
    RGB = np.array([int(alpha * 255), 0, int((1 - alpha) * 255)])
    return RGB

def ClosestLabel(binary, x, y):
    x = int(x)
    y = int(y)
    max_x , max_y = binary.shape[0:2] 
    x = max(0, min(max_x - 1, x))
    y = max(0, min(max_y - 1, y))
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
    try:
            return possible_labels[1]
    except:
        pdb.set_trace()
if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table",type="string",
                      help="table")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")
    parser.add_option("--key", dest="key",type="string",
                      help="patient id")
    parser.add_option("--marge", dest="marge", type="int",
              help="how much to reduce indexing")
    (options, args) = parser.parse_args()
    bin = label(imread("Job_{}/bin/{}".format(options.key, options.table.replace('.csv', '.tiff'))))
    max_rank = pd.read_csv('Job_{}/{}_whole_slide.csv'.format(options.key, options.key)).shape[0]
    CheckOrCreate(options.out)
    x, y = bin.shape
    table = pd.read_csv(options.table, header=0, index_col=0, sep=';')
    table = table.drop('coord_res_0', 1)
    table = table.drop('Parent', 1)
    table = table[table.notnull().all(axis=1)]
    if table.shape[0] == 0:
        for feat in [el for el in table.columns if "_rank" in el]:
            out = join(options.out, "feat_" + feat.split('_rank')[0])
            CheckOrCreate(out)
            color_copy = np.zeros(shape=(x,y,3), dtype='uint8')
            imsave(join(out, basename(options.table).replace("csv", "tiff")), color_copy, resolution=[1.0,1.0]) 
    else:
        for feat in [el for el in table.columns if "_rank" in el]:
            color_copy = np.zeros(shape=(x,y,3), dtype='uint8')
            def f(val, x, y):
        x -= options.marge + 2
        y -= options.marge + 2
                label = ClosestLabel(bin, x, y)
                color_copy[bin == label] = DivergingPurpleGreen(val, max_rank)

            table.apply(lambda row: f(row[feat], row["Centroid_x"], row["Centroid_y"]), axis=1)
            out = join(options.out, "feat_" + feat.split('_rank')[0])
            CheckOrCreate(out)
            imsave(join(out, basename(options.table).replace("csv", "tiff")), color_copy, resolution=[1.0,1.0])
