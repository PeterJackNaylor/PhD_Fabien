"""
TO DO: no proportion correction between samples.

"""
import pandas as pd
import glob as g
from optparse import OptionParser
from os.path import join
import numpy as np
import pdb
from numpy.random import shuffle

def myspecialshuffling(table, loops = 1):
    """
    Creates a training txt file
    """
    table = table.reset_index()
    table = table.drop("index", 1)
    lists = table["Label"]
    n1 = table["Label"].sum()
    n0 = table.shape[0] - n1
    table["Weight"] = 1
    table.loc[table["Label"] == 1, "Weight"] = float(n0) / float(n1)
    table = table.sample(n=table.shape[0]*loops, replace=True, weights="Weight", axis=0)
    table = table.reset_index()
    table = table.drop("index", 1)
    return table


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input", default="/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/NextFlow/Camelyon2016/OUTPUT",
                      help="input text folder")

    parser.add_option("-o", "--output", dest="output",
                      help="Where to the ouput txt files")
    parser.add_option("--split_value", dest="split_value",type="float",
                      help="Split ratio between number of images cancer/non cancer and nber of patient in test with normal or tumor")
    parser.add_option("--loops", dest="loops", type="int", default=1,
                    help="Number of loops of the biggest dataframe to perform")

    (options, args) = parser.parse_args()


    txt_loc = join(options.input, "*.txt")
    first = True
    for files in g.glob(txt_loc):

        colname = ['x', 'y', 'size_x', 'size_y', 'ref_level',
                         'DomainLabel', 'Label']
        table = pd.read_csv(files, sep = " ", header = None, 
                            names=colname, nrows=20)
        if first:
            result = table.copy()
            first = False
        else:
            result = result.append(table)
    patient = np.unique(result['DomainLabel'])
    TumorPatient = [f for f in patient if "Tumor" in f]
    NormalPatient = [f for f in patient if "Normal" in f]

    shuffle(TumorPatient)
    shuffle(NormalPatient)

    n_normal = len(NormalPatient)
    n_tumor = len(TumorPatient)

    test_tumor = TumorPatient[0:int(options.split_value * n_tumor)]
    test_normal = NormalPatient[0:int(options.split_value * n_normal)] 

    TumorTable = result[result['DomainLabel'].isin(test_tumor)].copy()
    NormalTable = result[result['DomainLabel'].isin(test_normal)].copy()
    
    TestTable = NormalTable.append(TumorTable)



    TrainTable = result.copy() 
    for pat_id in test_tumor + test_normal:
        TrainTable = TrainTable[TrainTable["DomainLabel"] != pat_id]

    TrainTable = myspecialshuffling(TrainTable, options.loops)
    TrainTable.to_csv(join(options.output,'train.txt'),sep = " ", header = None)

    TestTable = TestTable.sample(n=TestTable.shape[0], replace=False, axis=0)
    TestTable = TestTable.reset_index()
    TestTable = TestTable.drop("index", 1)
    TestTable["Weight"] = 1
    TestTable.to_csv(join(options.output,'test.txt'),sep = " ", header = None)