import numpy as np
from optparse import OptionParser
from GetStatistics4Color import list_f, CheckOrCreate
import glob
from os.path import join, basename
from skimage.measure import label
import pandas as pd
from WrittingTiff.Extractors import list_f_names
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--table", dest="table",type="string",
                      help="table")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")

    (options, args) = parser.parse_args()
    CheckOrCreate(options.out)

    whole_slide = pd.read_csv(options.table)

    for el in list_f_names:
        if el not in ["Centroid_x", "Centroid_y"]:
            sns.set(style="white", palette="muted", color_codes=True)
            f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
            sns.despine(left=True)
            sns.distplot(whole_slide[el], color="m", ax=axes)
            outname = join(options.out, el + ".png")
            f.savefig(outname)
