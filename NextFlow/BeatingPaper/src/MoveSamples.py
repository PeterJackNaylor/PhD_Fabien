import pandas as pd
import numpy as np
from UsefulFunctions.RandomUtils import CheckOrCreate
import os
from os.path import join

def MoveNEE(origin):
    os.rename(origin, join('./samples_NEE', origin))

table = pd.read_csv('Best_by_groups.csv', index_col=0)
CheckOrCreate('./samples_NEE')
val = np.unique(table["Model_Unique"])

for mod in val:
    if mod != 'Neeraj_PAPER':
        if 'fcn8s' in mod:
            orign_name = 'samples_FCN__' + mod
        elif '__UNet' in mod:
            orign_name = "samples_UNet__" + mod.split('__')[0]
        elif 'Dist' in mod:
            orign_name = "samples_Dist__" + mod

        MoveNEE(orign_name)
