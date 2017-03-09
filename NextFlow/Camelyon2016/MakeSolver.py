import numpy as np
from optparse import OptionParser
import os
import pdb
from Training.Solver import solver



if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--trainnet", dest="trainnet", help="train net path")
    parser.add_option("--testnet", dest="testnet", help="test net path")
    parser.add_option("--lr", dest="lr", type="float" ,help="learning rate")
    parser.add_option("--momentum", dest="momentum", type='float', help="momentum")
    parser.add_option("--weight_decay", dest="weight_decay", type="float", help="batch size")
    parser.add_option("--gamma", dest="gamma", type="float", help="gamma (stepsize reduction)")
    parser.add_option("--stepsize", "--stepsize", type="int", dest="stepsize",
                      help="stepsize value")

    (options, args) = parser.parse_args()
    solver("solver.prototxt", options.trainnet, options.testnet, options.lr, 
            options.momentum, options.weight_decay, options.gamma, options.stepsize)