from optparse import OptionParser
import caffe
from WrittingTiff.FrancoisRadvani import pred_f, crop
from Deprocessing.Transfer import ChangeEnv
import os
from WrittingTiff.createfold import GetNet
from scipy.misc import imread, imsave
import pdb
if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--file_name", dest="file_name",
                      help="Input file (raw data)")

    parser.add_option("--output", dest="output",
                      help="Where to store the outputs")
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
    net_1 = GetNet(cn_1, wd_1)
    net_2 = GetNet(cn_2, wd_1)

    base = os.path.basename(options.file_name).replace(".png", "")

    base_seg = os.path.join(options.output, base + "_seg.png")
    base_prob= os.path.join(options.output, base + "_prob.png")


    image = imread(options.file_name)[:,:,0:3]
    image = crop(image)
    image, prob = pred_f(image, net_1, net_2, stepSize=204, windowSize=(224, 224), param=7, border = 10)
    imsave(base_seg, image)
    imsave(base_prob, prob)


