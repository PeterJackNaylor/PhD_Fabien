import glob 
from scipy.misc import imsave
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import openslide as op
import pdb
import pandas as pd
from scipy.misc import imsave
from os.path import basename, join
from WrittingTiff.Extractors import list_f_names
from UsefulFunctions.RandomUtils import CheckOrCreate

## removing lines with 0


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--resolution", dest="res",type="int",
                      help="res")
    parser.add_option("--slide", dest="slide",type="string",
                      help="slide name")
    parser.add_option("--marge_cut_off", dest="marge", type="int",
                       help="int for the value to cut off on the sides of the tiff")
    (options, args) = parser.parse_args()
    patient = basename(options.slide).split('.')[0]
    slide = op.open_slide(options.slide)
    iter = glob.glob("*.npy")
    list_table = []

    for el in iter:
        tmp_table = pd.DataFrame(np.load(el), columns=list_f_names)
        tmp_table["Parent"] = el.split('.')[0]
        list_table.append(tmp_table)
    table = pd.concat(list_table)
    table = table.reset_index(drop=True)



    def Coordinates_0(x, y, parent):
        para = parent.split('_')
        x, y = int(x), int(y)
        X, Y = int(x) + int(para[1]), int(y) + int(para[2]) 
        return X, Y


    def InBox(x, y, parent):
        para = parent.split('_')
        width = int(para[4])
        height = int(para[3])
        if (x < options.marge) or (x > width - options.marge) or (y < options.marge) or (y > height - options.marge):
            return 0
        else:
            return 1

    table["coord_res_0"] = table.apply(lambda r: Coordinates_0(r['Centroid_x'], r['Centroid_y'], r['Parent']), axis=1)
    table["InBox"] = table.apply(lambda r: InBox(r['Centroid_x'], r['Centroid_y'], r['Parent']), axis=1)

    col_name = table.columns[:-4]
    table.ix[table['InBox']==0, col_name] = [0] * len(table.columns[:-4])
    table = table.drop("InBox", axis=1)
    ### killing doubles that overlap with two tiff
    group = table.groupby('coord_res_0')['Parent'].unique()
    index_to_iter = group[group.apply(lambda x: len(x)>1)]

    def CentroidInImage(x, y, parent):
        para = parent.split('_')
        width = int(para[4]) - 2 * options.marge
        height = int(para[3]) - 2 * options.marge
        x, y = int(x) - options.marge, int(y) - options.marge

        if x < 0 or x > width or y < 0 or y > height:
            return max(width, height)
        else:
            distance_to_border = [x - 0, y - 0, width - x, height - y]
            return min(distance_to_border)



    n_cols = len(table.columns[:-3])

    for el in index_to_iter.index:
        df = table[table["coord_res_0"] == el]
        df["distance_to_border"] = df.apply(lambda row: CentroidInImage(row['Centroid_x'], row['Centroid_y'], row['Parent']) ,axis=1)
        keep_index = df['distance_to_border'].idxmin()
        other_index = [el for el in df.index if el != keep_index]
        for other_el in other_index:
            table.iloc[other_el][table.columns[:-3]] = [0.] * n_cols
    without_parent = table.drop('Parent', 1)
    without_parent = without_parent.drop('coord_res_0', 1)
    without_parent = without_parent[(without_parent.T != 0).any()]





    table2 = table.ix[without_parent.index] 
    table2.to_csv("Job_{}/".format(patient) + '{}_whole_slide.csv'.format(patient))

    

    out = "Job_{}/feature_map/".format(patient)
    CheckOrCreate(out)
    names = []
    for feat in list_f_names:
        if feat not in ["Centroid_x", "Centroid_y", "coord_res_0"]:
            names.append(feat + "_rank")
            without_parent[feat + "_rank"] = without_parent[feat].rank(ascending=True)
            table2 = pd.concat([table2, without_parent[feat + "_rank"]], axis=1)

    output_new_tab = "Job_{}/RankedTable".format(patient)
    CheckOrCreate(output_new_tab)
    ranks = table2[names]
    result = pd.concat([table, ranks], axis=1, join_axes=[table.index])
    for group_name, df in result.groupby(['Parent']):
        with open(join(output_new_tab, group_name + ".csv"), 'w+') as f:
            df.to_csv(f, sep=';')
