import numpy as np
import pdb
import pandas as pd
from optparse import OptionParser
from os.path import join
from UsefulFunctions.RandomUtils import CheckOrCreate
from WrittingTiff.Extractors import list_f_names
from UsefulFunctions.UsefulOpenSlide import GetWholeImage, get_X_Y_from_0
from skimage.filters import gaussian
import openslide as op
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.misc import imsave

list_f = [np.min, np.mean, np.std, np.max]

def transparent_cmap(cmap, N=255, transparant = 0.5):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.ones(N + 4)
    mycmap._lut[:,-1] -= transparant
    mycmap._lut[0,-1] = 0 
    return mycmap

mycmap = transparent_cmap(plt.cm.jet)

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
    parser.add_option("--smooth", dest="smooth",type="int", default=5,
                      help="how much to smooth the feature map")
    (options, args) = parser.parse_args()

    name = options.table_name

    table = pd.read_csv(name)
    CheckOrCreate(options.out)
    slide = op.open_slide(options.slide)
    image = GetWholeImage(slide, level = options.res)
    x_S, y_S = image.size   
    #pdb.set_trace()
    def h(coord):
        x0, y0 = [int(el) for el in coord[1:-1].split(', ')]
        va, va2 = get_X_Y_from_0(slide, x0, y0, options.res)
        return va, va2
    feat_res = "coord_res_{}".format(options.res)
    table[feat_res] = table.apply(lambda r: h(r['coord_res_0']), axis=1)

    for el in list_f_names:
        if el not in ["Centroid_x", "Centroid_y"]:
            result = np.zeros(shape=(y_S, x_S))
            avg = np.zeros(shape=(y_S, x_S))

            def f(val, coord):
                indX, indY = coord
                result[int(indX), int(indY)] += val
                avg[int(indX), int(indY)] += 1

            table.apply(lambda row: f(row[el], row[feat_res]), axis=1)
            avg[avg == 0] += 1
            result = result / avg 


            combine_name = join(options.out, "{}.png").format(el)

            hm = gaussian(result, options.smooth)

            plt.imshow(np.array(image))
            plt.imshow(hm, cmap=mycmap)
            plt.colorbar()
	    plt.axis('off') 
            plt.savefig(combine_name, bbox_inches='tight')
            plt.clf()
