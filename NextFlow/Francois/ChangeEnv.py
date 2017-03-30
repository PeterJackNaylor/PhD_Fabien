from optparse import OptionParser
import caffe
from Deprocessing.Transfer import ChangeEnv
import os

if __name__ == "__main__":
    

    parser = OptionParser()
    parser.add_option('--env', dest="env", 
                      help="Where the annotated images are")
    parser.add_option('--wd', dest="wd", 
                      help="working directory")

    (options, args) = parser.parse_args()


    caffe.set_mode_cpu()

    cn_1 = "FCN_0.01_0.99_0.005"
    wd_1 = options.wd #"/share/data40T_v2/Peter/pretrained_models"
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
    wd_2 = wd_1

    ChangeEnv(options.env, os.path.join(wd_1, cn_1))
    ChangeEnv(options.env, os.path.join(wd_1, cn_2))
