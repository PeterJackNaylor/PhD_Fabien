import numpy as np
import pdb
import pandas as pd
from optparse import OptionParser
from os.path import join
from UsefulFunctions.RandomUtils import CheckOrCreate
from WrittingTiff.Extractors import list_f_names
from UsefulFunctions.UsefulOpenSlide import GetWholeImage, get_X_Y_from_0
import openslide as op
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.misc import imsave

list_f = [np.min, np.mean, np.std, np.max]

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table_name",type="string",
                      help="table name")
    parser.add_option("--output", dest="out", type="string",
                      help="output localisation")
    parser.add_option("--res", dest="res", type="int",
                      help="resolution to visualise the results")
    parser.add_option("--slide", dest="slide",type="string",
                      help="slide name")

    (options, args) = parser.parse_args()

    name = options.table_name

    table = pd.read_csv(name)
    CheckOrCreate(options.out)
    slide = op.open_slide(options.slide)
    image = GetWholeImage(slide, level = options.res)
    y_S, x_S = image.size   
    #pdb.set_trace()
    def h(coord):
        x0, y0 = [int(el) for el in coord[1:-1].split(', ')]
        va, va2 = get_X_Y_from_0(slide, x0, y0, options.res)
        return va, va2
    feat_res = "coord_res_{}".format(options.res)
    table[feat_res] = table.apply(lambda r: h(r['coord_res_0']), axis=1)

    for el in list_f_names:
        if el not in ["Centroid_x", "Centroid_y"]:
            result = np.zeros(shape=(x_S, y_S))
            avg = np.zeros(shape=(x_S, y_S))

            def f(val, coord):
                indX, indY = coord
                result[int(indX), int(indY)] += val
                avg[int(indX), int(indY)] += 1

            table.apply(lambda row: f(row[el], row[feat_res]), axis=1)
            avg[avg == 0] += 1
            result = result / avg 


            x, y = np.arange(0, y_S, 1.), np.arange(0, x_S, 1.)
            xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
            extent = xmin, xmax, ymin, ymax

            heat_name = join(options.out, "heat_map_{}.png").format(el)
            combine_name = join(options.out, "{}.png").format(el)
            rgb_name =  join(options.out, "RGB_{}.png").format(options.res)

            imsave(rgb_name, np.array(image))
            imsave(heat_name, result)

            fig = plt.figure(frameon=False)
            im1 = plt.imshow(np.array(image), extent=extent)
            im2 = plt.imshow(result, cmap=plt.cm.jet, alpha=.3, interpolation='bilinear',
                         extent=extent)
            fig.savefig(combine_name)
