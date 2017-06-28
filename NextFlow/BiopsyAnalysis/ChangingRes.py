import numpy as np
from optparse import OptionParser
import openslide as op
from UsefulFunctions.RandomUtils import CheckOrCreate
from WrittingTiff.Extractors import list_f_names
from UsefulFunctions.UsefulOpenSlide import get_X_Y_from_0
import pdb
import pandas as pd

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table_name",type="string",
                      help="table name")
    parser.add_option("--resolution", dest="res", type="int",
                      help="to visualize at res $res")
    parser.add_option("--slide", dest="slide", type="string",
                      help="to visualize at res $res")
    (options, args) = parser.parse_args()

    slide = op.open_slide(options.slide)
    table = pd.DataFrame(np.load(options.table_name), columns=list_f_names)
    res = options.res
    para = options.table_name.split('.')[0].split('_')

    def f(x, y):
        X, Y = int(x) + int(para[1]), int(y) + int(para[2])
        va, va2 = get_X_Y_from_0(slide, X, Y, res)
        return va, va2

    table["coord"] = table.apply(lambda row: f(row["Centroid_x"], row["Centroid_y"]), axis=1)
    folder = "Job_{}/tables_res_0/".format(para[0])
    CheckOrCreate(folder)
    
    table.to_csv(folder + options.table_name.replace('.npy', "_tables_res_0.csv"))
