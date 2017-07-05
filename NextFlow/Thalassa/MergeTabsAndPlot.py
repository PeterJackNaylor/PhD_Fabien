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

if __name__ == "__main__":


    parser = OptionParser()
    parser.add_option("--resolution", dest="res",type="int",
                      help="res")
    (options, args) = parser.parse_args()

    slide = op.open_slide('/share/data40T_v2/Peter/Data/Biopsy/579673.tiff')
    iter = glob.glob("*_general.csv")
    list_table = []
    for el in iter:
	list_table.append(pd.read_csv(el, index_col=0))
    table = pd.concat(list_table)
    table.to_csv('whole_slide.csv')
    image = GetWholeImage(slide, level = options.res)
    x, y = image.size
    result = np.zeros(shape=(x, y))
    avg = np.zeros(shape=(x, y))
    def f(val,coord):
#	pdb.set_trace()
	indX, indY = coord[1:-1].split(", ")
	result[int(indX), int(indY)] += val
	avg[int(indX), int(indY)] += 1
    table.apply(lambda row: f(row["sum"], row["coord"]), axis=1)
    avg[avg == 0] += 1
    result = result / avg 
    x, y = np.arange(0, x, 1.), np.arange(0, y, 1.)
    xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
    extent = xmin, xmax, ymin, ymax
    imsave("raw_rgb.png", np.array(image).transpose(1,0,2))
    imsave("heat_without.png", result)
    fig = plt.figure(frameon=False)
    im1 = plt.imshow(np.array(image).transpose(1,0,2), extent=extent)
    im2 = plt.imshow(result, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear',
                 extent=extent)
    fig.savefig('heatmap_overlay.png')
