import numpy as np
from optparse import OptionParser
import os
import pdb

from Nets.HalfNets import MakeHalfDeconvNet



if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--split", dest="split", help="split")
    parser.add_option("--dg", dest="dg", help="datagen adresse")
    parser.add_option("--cn", dest="cn", help="classifier name")
    parser.add_option("--loss", dest="loss", help="loss to specify")
    parser.add_option("--batch_size", dest="bs", type="int", help="batch size")
    parser.add_option("--num_output", dest="num_output", type="int", help="Number of outputs")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Where to store the patches")

    (options, args) = parser.parse_args()
    MakeHalfDeconvNet(options)