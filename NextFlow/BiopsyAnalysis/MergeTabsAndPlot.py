import glob 
from scipy.misc import imsave
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import openslide as op
from UsefulFunctions.UsefulOpenSlide import GetWholeImage, get_X_Y_from_0
import pdb
import pandas as pd
from scipy.misc import imsave
from os.path import basename, join
from WrittingTiff.Extractors import list_f_names
import pdb
from UsefulFunctions.RandomUtils import CheckOrCreate


## removing lines with 0


if __name__ == "__main__":

    list_f_names.append("coord")

    parser = OptionParser()
    parser.add_option("--resolution", dest="res",type="int",
                      help="res")
    parser.add_option("--slide", dest="slide",type="string",
                      help="slide name")
    (options, args) = parser.parse_args()
    patient = basename(options.slide).split('.')[0]
    slide = op.open_slide(options.slide)
    iter = glob.glob("*_tables_res_0.csv")
    list_table = []

    for el in iter:
        tmp_table = pd.read_csv(el, index_col=0)
        tmp_table["Parent"] = el.split('_tables_res_0')[0]
        list_table.append(tmp_table)

    table = pd.concat(list_table)
    table = table.reset_index(drop=True)
    without_parent = table.drop('Parent', 1)
    without_parent = without_parent.drop('coord', 1)

    without_parent = without_parent[(without_parent.T != 0).any()]
    table2 = table.ix[without_parent.index] 
    table2.to_csv("Job_{}/".format(patient) + '{}_whole_slide.csv'.format(patient))

    image = GetWholeImage(slide, level = options.res)
    x_S, y_S = image.size
    out = "Job_{}/feature_map/".format(patient)
    CheckOrCreate(out)
    for feat in list_f_names:
        if feat not in ["Centroid_x", "Centroid_y", "coord"]:

            without_parent[feat + "_rank"] = without_parent[feat].rank(ascending=True)
            table2 = pd.concat([table2, without_parent[feat + "_rank"]], axis=1)



            result = np.zeros(shape=(x_S, y_S))
            avg = np.zeros(shape=(x_S, y_S))
            def f(val,coord):
                indX, indY = coord[1:-1].split(", ")
                result[int(indX), int(indY)] += val
                avg[int(indX), int(indY)] += 1

            table2.apply(lambda row: f(row[feat], row["coord"]), axis=1)

            avg[avg == 0] += 1
            result = result / avg 

            x, y = np.arange(0, y_S, 1.), np.arange(0, x_S, 1.)
            xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
            extent = xmin, xmax, ymin, ymax

            heat_name = join(out, "heatmap_{}.png").format(feat)
            combine_name = join(out, "heatmap_with_RGB_{}.png").format(feat)
            rgb_name =  join(out, "RGB_{}.png").format(options.res)

            imsave(rgb_name, np.array(image).transpose(1,0,2))
            imsave(heat_name, result)

            fig = plt.figure(frameon=False)
            im1 = plt.imshow(np.array(image).transpose(1,0,2), extent=extent)
            im2 = plt.imshow(result, cmap=plt.cm.jet, alpha=.3, interpolation='bilinear',
                         extent=extent)
            fig.savefig(combine_name)

    output_new_tab = "Job_{}/RankedTable".format(patient)
    CheckOrCreate(output_new_tab)
    for group_name, df in table.groupby(['Parent']):
        with open(join(output_new_tab, group_name + ".csv"), 'a') as f:
            df.to_csv(f)
