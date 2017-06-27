import numpy as np
from optparse import OptionParser
import openslide as op
from UsefulFunctions.UsefulOpenSlide import get_X_Y_from_0
import pdb
import pandas as pd

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table_name",type="string",
                      help="table name")
    parser.add_option("--resolution", dest="res", type="int",
                      help="to visualize at res $res")
    (options, args) = parser.parse_args()

    slide = op.open_slide('/share/data40T_v2/Peter/Data/Biopsy/579673.tiff')

    table = pd.DataFrame(np.load(options.table_name), columns=["sum", "intens 0", "intens 5", "X", "Y"])
    res = options.res
    para = options.table_name.split('.')[0].split('_')

    def f(x, y):
    	X, Y = int(x) + int(para[0]), int(y) + int(para[1])
	va, va2 = get_X_Y_from_0(slide, X, Y, res)
    	return va, va2

    table["coord"] = table.apply(lambda row: f(row["X"], row["Y"]), axis=1)
    table.to_csv(options.table_name.replace('.npy', "_general.csv"))
