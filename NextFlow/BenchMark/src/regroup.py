#!/usr/bin/env python
import os
import pandas as pd
import pdb
from glob import glob
from optparse import OptionParser


parser = OptionParser()
parser.add_option('--store_best', dest='store_best',type='str')
parser.add_option('--output', dest="output", type="str")

(options, args) = parser.parse_args()

CSV = glob('*.csv')
df_list = []
for f in CSV:
    df = pd.read_csv(f, index_col=False)
    name = f.split('__')[0].split('/')[-1]
    df.index = [name]
    df_list.append(df)
table = pd.concat(df_list)
best_index = table['F1'].argmax()
table.to_csv(options.output)
tmove_name = "{}".format(best_index)
TOMOVE = glob(options.output.split('_')[0] + "*")
n_feat = options.output.split('_')[3]
name = 'best_model'
os.mkdir(name)
for file in TOMOVE:
    os.rename(file, os.path.join(name, file))