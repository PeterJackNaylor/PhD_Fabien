import numpy as np
from optparse import OptionParser
import os
import pdb
from Training.TrainModel import train



if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--solver", dest="solver", help="solver path")
    parser.add_option("--weight", dest="weight", default = None, help="weight path")
    parser.add_option("--wd", dest="wd" ,help="where to save the files")
    parser.add_option("--cn", dest="cn", help="working directory name")
    parser.add_option("--n_iter", dest="n_iter", type="int", help="number of iteration")
    parser.add_option("--disp_interval", dest="disp_interval", type="int",
                      help="display every such value")
    parser.add_option("--number_of_test", type="int", dest="number_of_test",
                      help="number of times to loop over test set")
    parser.add_option("--num", dest="num", help="to add to the name")

    (options, args) = parser.parse_args()
    train(options.solver, options.weight, options.wd, options.cn, 
            options.n_iter, options.disp_interval, options.number_of_test, options.num)