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
    parser.add_option("--slide", dest="slide",type="string",
                      help="slide name")
    parser.add_option("--marge", dest="marge", type="int",
                       help="int for the value to cut off on the sides of the tiff")
    (options, args) = parser.parse_args()
    patient = basename(options.slide).split('.')[0]
    slide = op.open_slide(options.slide)
    iter = glob.glob("*.csv")
    list_table = []

    for el in iter:
        tmp_table = pd.read_csv(el, index_col=0)
        na = el.split('.')[0].split('_')[1::]
        tmp_table["Parent"] = '_'.join(na)
        list_table.append(tmp_table)
    table = pd.concat(list_table)
    table = table.reset_index(drop=True)
    table["zeros"] = 0
    table.ix[table['Pixel_sum']==0, 'zeros'] = 1

    def Coordinates_0(x, y, parent):
        para = parent.split('_')
        X, Y = int(x) + int(para[2]), int(y) + int(para[1]) 
        return X, Y
    def distancefromborder(x, y, parent):
        para = parent.split('_')
        width = int(para[4])
        height = int(para[3])
        dist_ = [x, y, width - x, height - y]
        return min(dist_)
    table["coord_res_0"] = table.apply(lambda r: Coordinates_0(r['Centroid_x'], r['Centroid_y'], r['Parent']), axis=1)
    table["dupl"] = 0
    table["id_dupl"] = 0
    grouped = table.groupby(['coord_res_0'])
    dupli = grouped.filter(lambda x: len(x) > 1)
    for el in np.unique(dupli['coord_res_0']):
        small = dupli[dupli['coord_res_0']==el]
        small['distancefromborder'] = small.apply(lambda r: distancefromborder(r['Centroid_x'], r['Centroid_y'], r['Parent']), axis=1)
        small['postoborder'] = small['distancefromborder'].rank(ascending=False)
        keep = small.index[0]
        dupl = small.index[1::]
        for el in dupl:
            table.ix[el, "dupl"] = 1
            table.ix[el, "id_dupl"] = keep
    without_dup = table.copy()
    without_dup = without_dup[without_dup["dupl"] == 0]
    without_dup = without_dup[without_dup['Pixel_sum'] != 0]
   
    without_dup.to_csv('{}.csv'.format(patient))

    

    out = "feature_map_{}/".format(patient)
    CheckOrCreate(out)
    names = []
    for feat in list_f_names:
        if feat not in ["Centroid_x", "Centroid_y", "coord_res_0"]:
            names.append(feat + "_rank")
            without_dup[feat + "_rank"] = without_dup[feat].rank(ascending=True)
    def giveranktodupl(row):
        if row["zeros"] == 1:
            return [1.] * len(names)    
        if row["dupl"] == 1:
            return without_dup.ix[row['id_dupl'], names]
        else:
            return without_dup.ix[row.name, names]
    table[names] = table.apply(lambda row: giveranktodupl(row), axis=1)
    output_new_tab = "Ranked_{}_{}".format(patient, feat)
    CheckOrCreate(output_new_tab)
    table.drop('dupl', axis=1, inplace=True)
    table.drop('id_dupl', axis=1, inplace=True)
    table.drop('zeros', axis=1, inplace=True)
    for group_name, df in table.groupby(['Parent']):
        with open(join(output_new_tab, "ranked_" + group_name + ".csv"), 'w+') as f:
            df.to_csv(f, sep=';')
