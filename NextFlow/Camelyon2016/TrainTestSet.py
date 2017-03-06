"""
TO DO: no proportion correction between samples.

"""
import pandas as pd
import glob as g
from optparse import OptionParser
from os.path import join
import numpy as np
from numpy.random import shuffle
if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input",
                      help="input text folder")

    parser.add_option("-o", "--output", dest="output",
                      help="Where to the ouput txt files")

    parser.add_option("--nber_patient", dest="nber_patient",type='int',
                      help="Number of patients to put in test/train set")

    parser.add_option("--split_value", dest="split_value",type="float",
                      help="Split ratio between number of images cancer/non cancer and nber of patient in test with normal or tumor")

    (options, args) = parser.parse_args()


    txt_loc = join(options.input, "*.txt")
    first = True
    for files in g.glob(txt_loc):
        table = pd.read_csv(files, sep = " ", header = None)
        table.columns = ["id", 'x', 'y', 'size_x', 'size_y', 'ref_level',
                         'DomainLabel', 'Label']
        if first:
            result = table.copy()
        else:
            result = result.append(table)
    
    patient = np.unique(result['DomainLabel'])
    TumorPatient = [f in patient if "Tumor" in patient]
    NormalPatient = [f in patient if "Normal" in patient]

    shuffle(TumorPatient)
    shuffle(NormalPatient)


    test_tumor = TumorPatient[0:(options.split_value*options.nber_patient)]
    test_normal = NormalPatient[0:((1 - options.split_value)*options.nber_patient)] 

    TumorTable = result[result['DomainLabel'].isin(test_tumor)].copy()
    NormalTable = result[result['DomainLabel'].isin(test_normal)].copy()
    
    TestTable = NormalTable.append(TumorTable)



    TrainTable = result.copy() 
    for pat_id in test_tumor + test_normal:
        TrainTable = TrainTable[TrainTable != pat_id]

    TrainTable.to_csv(join(options.output,'train.txt'),sep = " ", header = None)
    TestTable.to_csv(join(options.output,'test.txt'),sep = " ", header = None)